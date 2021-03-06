/**
 * Proper ethash OpenCL kernel compatible with AMD and NVIDIA
 *
 * (c) tpruvot @ October 2016
 */


#ifndef WORKSIZE
#define WORKSIZE 256
#define COMPILE_MAIN_ONLY
#define PRECALC_BLAKE
#endif

#ifndef LYRA_WORKSIZE
#define LYRA_WORKSIZE WORKSIZE
//WORKSIZE
#endif

#ifndef MAX_GLOBAL_THREADS
#define MAX_GLOBAL_THREADS 64
#endif


#ifndef TEST_KERNEL_HASH
#define BMW32_ONLY_RETURN_LAST
#endif

#define SPH_COMPACT_BLAKE_64 0

#define ACCESSES   64
#define MAX_OUTPUTS 255u
#define barrier(x) mem_fence(x)

#define WORD_BYTES 4
#define DATASET_BYTES_INIT 536870912
#define DATASET_BYTES_GROWTH 12582912
#define CACHE_BYTES_INIT 8388608
#define CACHE_BYTES_GROWTH 196608
#define EPOCH_LENGTH 400
#define CACHE_MULTIPLIER 64
#define MIX_BYTES 64
#define HASH_BYTES 32
#define DATASET_PARENTS 256
#define CACHE_ROUNDS 3
#define ACCESSES 64
#define FNV_PRIME 0x01000193U

#define MAX_NONCE_OUTPUTS 255
#define MAX_HASH_OUTPUTS  256

// DAG Cache node
typedef union _Node
{
	uint dwords[8];
	uint4 dqwords[2];
} Node; // NOTE: should be HASH_BYTES long


typedef union _MixNodes {
	uint values[16];
	uint8 nodes8[2];
	uint16 nodes16;
} MixNodes;


// Output hash
typedef union {
	uint h4[8];
	ulong h8[4];
	uint2 u2[4];
} hash32_t;

//#define fnv(x, y) ((x) * FNV_PRIME ^ (y)) % (0xffffffff)
//#define fnv_reduce(v) fnv(fnv(fnv(v.x, v.y), v.z), v.w)

inline uint fnv(const uint v1, const uint v2) {
	return ((v1 * FNV_PRIME) ^ v2) % (0xffffffff);
}

inline uint4 fnv4(const uint4 v1, const uint4 v2) {
	return ((v1 * FNV_PRIME) ^ v2) % (0xffffffff);
}

#ifdef cl_nv_pragma_unroll
#define NVIDIA
#else
#pragma OPENCL EXTENSION cl_amd_media_ops2 : enable
#define AMD
#define ROTL64_1(x, y)  amd_bitalign((x), (x).s10, (32U - y))
#define ROTL64_2(x, y)  amd_bitalign((x).s10, (x), (32U - y))
#define ROTL64_8(x, y)  amd_bitalign((x), (x).s10, 24U)


#define ROTR64_8(x, y)  amd_bitalign((x), (x).s10, 24U)
#define BFE(x, start, len)  amd_bfe(x, start, len)

#endif

#ifdef NVIDIA
static inline uint2 rol2(const uint2 a, const uint offset) {
	uint2 r;
	asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(r.x) : "r"(a.y), "r"(a.x), "r"(offset));
	asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(r.y) : "r"(a.x), "r"(a.y), "r"(offset));
	return r;
}
static inline uint2 ror2(const uint2 a, const uint offset) {
	uint2 r;
	asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(r.x) : "r"(a.x), "r"(a.y), "r"(offset));
	asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(r.y) : "r"(a.y), "r"(a.x), "r"(offset));
	return r;
}
static inline uint2 rol8(const uint2 a) {
	uint2 r;
	asm("prmt.b32 %0, %1, %2, 0x6543;" : "=r"(r.x) : "r"(a.y), "r"(a.x));
	asm("prmt.b32 %0, %1, %2, 0x2107;" : "=r"(r.y) : "r"(a.y), "r"(a.x));
	return r;
}

#define ROTL64_1(x, y) rol2(x, y)
#define ROTL64_2(x, y) ror2(x, (32U - y))
#define ROTL64_8(x, y) rol8(x)
#define ROTR64_8(x, y) ror8(x)

static inline uint nv_bfe(const uint a, const uint start, const uint len) {
	uint r;
	asm("bfe.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(start), "r"(len));
	return r;
}
#define BFE(x, start, len) nv_bfe(x, start, len)
#endif /* NVIDIA */

//
// BEGIN BLAKE256
//

// Blake256 Macros
__constant static const uint sigma[16][16] = {
	{  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15 },
	{ 14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3 },
	{ 11,  8, 12,  0,  5,  2, 15, 13, 10, 14,  3,  6,  7,  1,  9,  4 },
	{  7,  9,  3,  1, 13, 12, 11, 14,  2,  6,  5, 10,  4,  0, 15,  8 },
	{  9,  0,  5,  7,  2,  4, 10, 15, 14,  1, 11, 12,  6,  8,  3, 13 },
	{  2, 12,  6, 10,  0, 11,  8,  3,  4, 13,  7,  5, 15, 14,  1,  9 },
	{ 12,  5,  1, 15, 14, 13,  4, 10,  0,  7,  6,  3,  9,  2,  8, 11 },
	{ 13, 11,  7, 14, 12,  1,  3,  9,  5,  0, 15,  4,  8,  6,  2, 10 },
	{  6, 15, 14,  9, 11,  3,  0,  8, 12,  2, 13,  7,  1,  4, 10,  5 },
	{ 10,  2,  8,  4,  7,  6,  1,  5, 15, 11,  9, 14,  3, 12, 13,  0 },
	{  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15 },
	{ 14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3 },
	{ 11,  8, 12,  0,  5,  2, 15, 13, 10, 14,  3,  6,  7,  1,  9,  4 },
	{  7,  9,  3,  1, 13, 12, 11, 14,  2,  6,  5, 10,  4,  0, 15,  8 },
	{  9,  0,  5,  7,  2,  4, 10, 15, 14,  1, 11, 12,  6,  8,  3, 13 },
	{  2, 12,  6, 10,  0, 11,  8,  3,  4, 13,  7,  5, 15, 14,  1,  9 }
};

__constant static const uint  c_u256[16] = {
	0x243F6A88, 0x85A308D3,
	0x13198A2E, 0x03707344,
	0xA4093822, 0x299F31D0,
	0x082EFA98, 0xEC4E6C89,
	0x452821E6, 0x38D01377,
	0xBE5466CF, 0x34E90C6C,
	0xC0AC29B7, 0xC97C50DD,
	0x3F84D5B5, 0xB5470917
};

#define SPH_C32(x)    ((uint)(x ## U))
#define SPH_T32(x) (as_uint(x))
#define SPH_ROTL32(x, n) rotate(as_uint(x), as_uint(n))
#define SPH_ROTR32(x, n)   SPH_ROTL32(x, (32 - (n)))

#define SPH_C64(x)    ((ulong)(x ## UL))
#define SPH_T64(x) (as_ulong(x))
#define SPH_ROTL64(x, n) rotate(as_ulong(x), (n) & 0xFFFFFFFFFFFFFFFFUL)
#define SPH_ROTR64(x, n)   SPH_ROTL64(x, (64 - (n)))


inline uint sph_bswap32(uint n) { return (rotate(n & 0x00FF00FF, 24U)|(rotate(n, 8U) & 0x00FF00FF)); }

#define BLAKE256_GS(m0, m1, c0, c1, a, b, c, d)   do { \
		a = SPH_T32(a + b + (m0 ^ c1)); \
		d = SPH_ROTR32(d ^ a, 16); \
		c = SPH_T32(c + d); \
		b = SPH_ROTR32(b ^ c, 12); \
		a = SPH_T32(a + b + (m1 ^ c0)); \
		d = SPH_ROTR32(d ^ a, 8); \
		c = SPH_T32(c + d); \
		b = SPH_ROTR32(b ^ c, 7); \
	} while (0)


#define BLAKE256_GS_ALT(a,b,c,d,x) { \
	const uint idx1 = sigma[R][x]; \
	const uint idx2 = sigma[R][x+1]; \
	V[a] += (M[idx1] ^ c_u256[idx2]) + V[b]; \
	V[d] ^= V[a]; \
    V[d] = SPH_ROTR32(V[d], 16); \
	V[c] += V[d]; \
    V[b] ^= V[c]; \
	V[b] = SPH_ROTR32(V[b], 12); \
\
	V[a] += (M[idx2] ^ c_u256[idx1]) + V[b]; \
    V[d] ^= V[a]; \
	V[d] = SPH_ROTR32(V[d], 8); \
	V[c] += V[d]; \
    V[b] ^= V[c]; \
	V[b] = SPH_ROTR32(V[b], 7); \
}

#define BLAKE256_STATE \
uint H0, H1, H2, H3, H4, H5, H6, H7, T0, T1;
#define INIT_BLAKE256_STATE \
H0 = SPH_C32(0x6a09e667); \
H1 = SPH_C32(0xbb67ae85); \
H2 = SPH_C32(0x3c6ef372); \
H3 = SPH_C32(0xa54ff53a); \
H4 = SPH_C32(0x510e527f); \
H5 = SPH_C32(0x9b05688c); \
H6 = SPH_C32(0x1f83d9ab); \
H7 = SPH_C32(0x5be0cd19); \
T0 = 0; \
T1 = 0;

#define BLAKE32_ROUNDS 14

#define BLAKE256_COMPRESS32_STATE \
uint M[16]; \
uint V[16]; \

#define BLAKE256_COMPRESS_BEGIN(b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15) \
V[0] = H0; \
V[1] = H1; \
V[2] = H2; \
V[3] = H3; \
V[4] = H4; \
V[5] = H5; \
V[6] = H6; \
V[7] = H7; \
V[8] = c_u256[0]; \
V[9] = c_u256[1]; \
V[10] = c_u256[2]; \
V[11] = c_u256[3]; \
V[12] = T0 ^ c_u256[4]; \
V[13] = T0 ^ c_u256[5]; \
V[14] = T1 ^ c_u256[6]; \
V[15] = T1 ^ c_u256[7]; \
M[0x0] = b0; \
M[0x1] = b1; \
M[0x2] = b2; \
M[0x3] = b3; \
M[0x4] = b4; \
M[0x5] = b5; \
M[0x6] = b6; \
M[0x7] = b7; \
M[0x8] = b8; \
M[0x9] = b9; \
M[0xA] = b10; \
M[0xB] = b11; \
M[0xC] = b12; \
M[0xD] = b13; \
M[0xE] = b14; \
M[0xF] = b15; \

#define BLAKE256_COMPRESS_BEGIN_DIRECT(H0, H1, H2, H3, H4, H5, H6, H7, T0, T1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15) \
V[0] = H0; \
V[1] = H1; \
V[2] = H2; \
V[3] = H3; \
V[4] = H4; \
V[5] = H5; \
V[6] = H6; \
V[7] = H7; \
V[8] = c_u256[0]; \
V[9] = c_u256[1]; \
V[10] = c_u256[2]; \
V[11] = c_u256[3]; \
V[12] = T0 ^ c_u256[4]; \
V[13] = T0 ^ c_u256[5]; \
V[14] = T1 ^ c_u256[6]; \
V[15] = T1 ^ c_u256[7]; \
M[0x0] = b0; \
M[0x1] = b1; \
M[0x2] = b2; \
M[0x3] = b3; \
M[0x4] = b4; \
M[0x5] = b5; \
M[0x6] = b6; \
M[0x7] = b7; \
M[0x8] = b8; \
M[0x9] = b9; \
M[0xA] = b10; \
M[0xB] = b11; \
M[0xC] = b12; \
M[0xD] = b13; \
M[0xE] = b14; \
M[0xF] = b15;

// Begin compress state with standard blake values from INIT_BLAKE_STATE
#define BLAKE256_COMPRESS_BEGIN_LIGHT(T0, T1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15) \
BLAKE256_COMPRESS_BEGIN_DIRECT(SPH_C32(0x6a09e667), SPH_C32(0xbb67ae85), SPH_C32(0x3c6ef372), SPH_C32(0xa54ff53a),  SPH_C32(0x510e527f), SPH_C32(0x9b05688c), SPH_C32(0x1f83d9ab), SPH_C32(0x5be0cd19), T0, T1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15)

#define BLAKE256_COMPRESS_END \
H0 ^= V[0] ^ V[8]; \
H1 ^= V[1] ^ V[9]; \
H2 ^= V[2] ^ V[10]; \
H3 ^= V[3] ^ V[11]; \
H4 ^= V[4] ^ V[12]; \
H5 ^= V[5] ^ V[13]; \
H6 ^= V[6] ^ V[14]; \
H7 ^= V[7] ^ V[15];

#define BLAKE256_COMPRESS_END_DIRECT(H0, H1, H2, H3, H4, H5, H6, H7, OUT_H0, OUT_H1, OUT_H2, OUT_H3, OUT_H4, OUT_H5, OUT_H6, OUT_H7) \
OUT_H0 = sph_bswap32(H0 ^ (V[0] ^ V[8])); \
OUT_H1 = sph_bswap32(H1 ^ (V[1] ^ V[9])); \
OUT_H2 = sph_bswap32(H2 ^ (V[2] ^ V[10])); \
OUT_H3 = sph_bswap32(H3 ^ (V[3] ^ V[11])); \
OUT_H4 = sph_bswap32(H4 ^ (V[4] ^ V[12])); \
OUT_H5 = sph_bswap32(H5 ^ (V[5] ^ V[13])); \
OUT_H6 = sph_bswap32(H6 ^ (V[6] ^ V[14])); \
OUT_H7 = sph_bswap32(H7 ^ (V[7] ^ V[15]));

#define BLAKE256_COMPRESS_END_DIRECT_NOSWAP(H0, H1, H2, H3, H4, H5, H6, H7, OUT_H0, OUT_H1, OUT_H2, OUT_H3, OUT_H4, OUT_H5, OUT_H6, OUT_H7) \
OUT_H0 = (H0 ^ (V[0] ^ V[8])); \
OUT_H1 = (H1 ^ (V[1] ^ V[9])); \
OUT_H2 = (H2 ^ (V[2] ^ V[10])); \
OUT_H3 = (H3 ^ (V[3] ^ V[11])); \
OUT_H4 = (H4 ^ (V[4] ^ V[12])); \
OUT_H5 = (H5 ^ (V[5] ^ V[13])); \
OUT_H6 = (H6 ^ (V[6] ^ V[14])); \
OUT_H7 = (H7 ^ (V[7] ^ V[15]));

#define BLAKE256_COMPRESS_END_LIGHT(OUT_H0, OUT_H1, OUT_H2, OUT_H3, OUT_H4, OUT_H5, OUT_H6, OUT_H7) \
BLAKE256_COMPRESS_END_DIRECT(SPH_C32(0x6a09e667), SPH_C32(0xbb67ae85), SPH_C32(0x3c6ef372), SPH_C32(0xa54ff53a),  SPH_C32(0x510e527f), SPH_C32(0x9b05688c), SPH_C32(0x1f83d9ab), SPH_C32(0x5be0cd19), OUT_H0, OUT_H1, OUT_H2, OUT_H3, OUT_H4, OUT_H5, OUT_H6, OUT_H7)

#define BLAKE256_COMPRESS_END_LIGHT_NOSWAP(OUT_H0, OUT_H1, OUT_H2, OUT_H3, OUT_H4, OUT_H5, OUT_H6, OUT_H7) \
BLAKE256_COMPRESS_END_DIRECT_NOSWAP(SPH_C32(0x6a09e667), SPH_C32(0xbb67ae85), SPH_C32(0x3c6ef372), SPH_C32(0xa54ff53a),  SPH_C32(0x510e527f), SPH_C32(0x9b05688c), SPH_C32(0x1f83d9ab), SPH_C32(0x5be0cd19), OUT_H0, OUT_H1, OUT_H2, OUT_H3, OUT_H4, OUT_H5, OUT_H6, OUT_H7)

#define BLAKE256_COMPRESS32(b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15) \
V[0] = H0; \
V[1] = H1; \
V[2] = H2; \
V[3] = H3; \
V[4] = H4; \
V[5] = H5; \
V[6] = H6; \
V[7] = H7; \
V[8] = c_u256[0]; \
V[9] = c_u256[1]; \
V[10] = c_u256[2]; \
V[11] = c_u256[3]; \
V[12] = T0 ^ c_u256[4]; \
V[13] = T0 ^ c_u256[5]; \
V[14] = T1 ^ c_u256[6]; \
V[15] = T1 ^ c_u256[7]; \
M[0x0] = b0; \
M[0x1] = b1; \
M[0x2] = b2; \
M[0x3] = b3; \
M[0x4] = b4; \
M[0x5] = b5; \
M[0x6] = b6; \
M[0x7] = b7; \
M[0x8] = b8; \
M[0x9] = b9; \
M[0xA] = b10; \
M[0xB] = b11; \
M[0xC] = b12; \
M[0xD] = b13; \
M[0xE] = b14; \
M[0xF] = b15; \
for (uint R=0; R< BLAKE32_ROUNDS; R++) { \
		BLAKE256_GS_ALT(0, 4, 0x8, 0xC, 0x0); \
		BLAKE256_GS_ALT(1, 5, 0x9, 0xD, 0x2); \
		BLAKE256_GS_ALT(2, 6, 0xA, 0xE, 0x4); \
		BLAKE256_GS_ALT(3, 7, 0xB, 0xF, 0x6); \
		BLAKE256_GS_ALT(0, 5, 0xA, 0xF, 0x8); \
		BLAKE256_GS_ALT(1, 6, 0xB, 0xC, 0xA); \
		BLAKE256_GS_ALT(2, 7, 0x8, 0xD, 0xC); \
		BLAKE256_GS_ALT(3, 4, 0x9, 0xE, 0xE); \
} \
H0 ^= V[0] ^ V[8]; \
H1 ^= V[1] ^ V[9]; \
H2 ^= V[2] ^ V[10]; \
H3 ^= V[3] ^ V[11]; \
H4 ^= V[4] ^ V[12]; \
H5 ^= V[5] ^ V[13]; \
H6 ^= V[6] ^ V[14]; \
H7 ^= V[7] ^ V[15];

//
// END BLAKE256
//


//
// BEGIN KECCAK32
//

__constant static const ulong RC[] = {
  SPH_C64(0x0000000000000001), SPH_C64(0x0000000000008082),
  SPH_C64(0x800000000000808A), SPH_C64(0x8000000080008000),
  SPH_C64(0x000000000000808B), SPH_C64(0x0000000080000001),
  SPH_C64(0x8000000080008081), SPH_C64(0x8000000000008009),
  SPH_C64(0x000000000000008A), SPH_C64(0x0000000000000088),
  SPH_C64(0x0000000080008009), SPH_C64(0x000000008000000A),
  SPH_C64(0x000000008000808B), SPH_C64(0x800000000000008B),
  SPH_C64(0x8000000000008089), SPH_C64(0x8000000000008003),
  SPH_C64(0x8000000000008002), SPH_C64(0x8000000000000080),
  SPH_C64(0x000000000000800A), SPH_C64(0x800000008000000A),
  SPH_C64(0x8000000080008081), SPH_C64(0x8000000000008080),
  SPH_C64(0x0000000080000001), SPH_C64(0x8000000080008008)
};

__constant static const uint2 keccak_round_constants35[24] = {
	{ 0x00000001ul, 0x00000000 }, { 0x00008082ul, 0x00000000 },
	{ 0x0000808aul, 0x80000000 }, { 0x80008000ul, 0x80000000 },
	{ 0x0000808bul, 0x00000000 }, { 0x80000001ul, 0x00000000 },
	{ 0x80008081ul, 0x80000000 }, { 0x00008009ul, 0x80000000 },
	{ 0x0000008aul, 0x00000000 }, { 0x00000088ul, 0x00000000 },
	{ 0x80008009ul, 0x00000000 }, { 0x8000000aul, 0x00000000 },
	{ 0x8000808bul, 0x00000000 }, { 0x0000008bul, 0x80000000 },
	{ 0x00008089ul, 0x80000000 }, { 0x00008003ul, 0x80000000 },
	{ 0x00008002ul, 0x80000000 }, { 0x00000080ul, 0x80000000 },
	{ 0x0000800aul, 0x00000000 }, { 0x8000000aul, 0x80000000 },
	{ 0x80008081ul, 0x80000000 }, { 0x00008080ul, 0x80000000 },
	{ 0x80000001ul, 0x00000000 }, { 0x80008008ul, 0x80000000 }
};
 
// SPH_ROTL64
//#define KROL2(x, y) as_uint2(rotate(as_ulong(x), (ulong)(y)) & 0xFFFFFFFFFFFFFFFFUL)

// SPH_ROTL64(s[19], 8);
//#define KROL8(x) as_uint2(rotate(as_ulong(x),  (ulong)(8)) & 0xFFFFFFFFFFFFFFFFUL)

// SPH_ROTL64(s[23], 56);     // 64-56=8
//#define KROR8(x) as_uint2(rotate(as_ulong(x),  (ulong)(56)) & 0xFFFFFFFFFFFFFFFFUL)

#if defined(AMD)

inline uint2 ROL2(const uint2 v, const int n)
{
	uint2 result;
	if (n <= 32)
	{
		return amd_bitalign((v).xy, (v).yx, 32 - n);
	}
	else
	{
		return amd_bitalign((v).yx, (v).xy, 64 - n);
	}
	return result;
}

#define KROL2(vv, r) ROL2(vv,r)

#else

inline uint2 ROL2(const uint2 v, const int n)
{
	uint2 result;
	if (n <= 32)
	{
		result.y = ((v.y << (n)) | (v.x >> (32 - n)));
		result.x = ((v.x << (n)) | (v.y >> (32 - n)));
	}
	else
	{
		result.y = ((v.x << (n - 32)) | (v.y >> (64 - n)));
		result.x = ((v.y << (n - 32)) | (v.x >> (64 - n)));
	}
	return result;
}

#define KROL2(vv, r) ROL2(vv, r)

#endif

// SPH_ROTL64(s[19], 8);
#define KROL8(x) KROL2(x, 8)
#define KROR8(x) KROL2(x, 56)

inline void keccak_block_uint2(uint2* s, const uint isolate)
{
	uint2 bc[5], tmpxor[5], u, v;
	//	uint2 s[25];
	
	#pragma nounroll
	for (int i = 0; i < 24; i++)
	{
		//if (isolate) {
		{
		#pragma unroll
		for (uint x = 0; x < 5; x++)
			tmpxor[x] = s[x] ^ s[x + 5] ^ s[x + 10] ^ s[x + 15] ^ s[x + 20];

		bc[0] = tmpxor[0] ^ KROL2(tmpxor[2], 1);
		bc[1] = tmpxor[1] ^ KROL2(tmpxor[3], 1);
		bc[2] = tmpxor[2] ^ KROL2(tmpxor[4], 1);
		bc[3] = tmpxor[3] ^ KROL2(tmpxor[0], 1);
		bc[4] = tmpxor[4] ^ KROL2(tmpxor[1], 1);

		u = s[1] ^ bc[0];

		s[0] ^= bc[4];
		s[1] = KROL2(s[6] ^ bc[0], 44);    // ROR2(a, 64 - offset(44)); 
		s[6] = KROL2(s[9] ^ bc[3], 20);
		s[9] = KROL2(s[22] ^ bc[1], 61);
		s[22] = KROL2(s[14] ^ bc[3], 39);
		s[14] = KROL2(s[20] ^ bc[4], 18);
		s[20] = KROL2(s[2] ^ bc[1], 62);
		s[2] = KROL2(s[12] ^ bc[1], 43);
		s[12] = KROL2(s[13] ^ bc[2], 25);
		s[13] = KROL8(s[19] ^ bc[3]);
		s[19] = KROR8(s[23] ^ bc[2]);
		s[23] = KROL2(s[15] ^ bc[4], 41);
		s[15] = KROL2(s[4] ^ bc[3], 27);
		s[4] = KROL2(s[24] ^ bc[3], 14);
		s[24] = KROL2(s[21] ^ bc[0], 2);
		s[21] = KROL2(s[8] ^ bc[2], 55);
		s[8] = KROL2(s[16] ^ bc[0], 45);
		s[16] = KROL2(s[5] ^ bc[4], 36);
		s[5] = KROL2(s[3] ^ bc[2], 28);
		s[3] = KROL2(s[18] ^ bc[2], 21);
		s[18] = KROL2(s[17] ^ bc[1], 15);
		s[17] = KROL2(s[11] ^ bc[0], 10);
		s[11] = KROL2(s[7] ^ bc[1], 6);
		s[7] = KROL2(s[10] ^ bc[4], 3);
		s[10] = KROL2(u, 1);
/*
		u = s[0]; v = s[1]; s[0] ^= (~v) & s[2]; s[1] ^= (~s[2]) & s[3]; s[2] ^= (~s[3]) & s[4]; s[3] ^= (~s[4]) & u; s[4] ^= (~u) & v;
		u = s[5]; v = s[6]; s[5] ^= (~v) & s[7]; s[6] ^= (~s[7]) & s[8]; s[7] ^= (~s[8]) & s[9]; s[8] ^= (~s[9]) & u; s[9] ^= (~u) & v;
		u = s[10]; v = s[11]; s[10] ^= (~v) & s[12]; s[11] ^= (~s[12]) & s[13]; s[12] ^= (~s[13]) & s[14]; s[13] ^= (~s[14]) & u; s[14] ^= (~u) & v;
		u = s[15]; v = s[16]; s[15] ^= (~v) & s[17]; s[16] ^= (~s[17]) & s[18]; s[17] ^= (~s[18]) & s[19]; s[18] ^= (~s[19]) & u; s[19] ^= (~u) & v;
		u = s[20]; v = s[21]; s[20] ^= (~v) & s[22]; s[21] ^= (~s[22]) & s[23]; s[22] ^= (~s[23]) & s[24]; s[23] ^= (~s[24]) & u; s[24] ^= (~u) & v;*/
		
		u = s[0]; v = s[1]; s[0] = bitselect(s[0]^s[2],s[0],s[1]); s[1] = bitselect(s[1]^s[3],s[1],s[2]); s[2] = bitselect(s[2]^s[4],s[2],s[3]); s[3] = bitselect(s[3]^u,s[3],s[4]); s[4] = bitselect(s[4]^v,s[4],u);
		u = s[5]; v = s[6]; s[5] = bitselect(s[5]^s[7],s[5],s[6]); s[6] = bitselect(s[6]^s[8],s[6],s[7]); s[7] = bitselect(s[7]^s[9],s[7],s[8]); s[8] = bitselect(s[8]^u,s[8],s[9]); s[9] = bitselect(s[9]^v,s[9],u);
	  u = s[10]; v = s[11]; s[10] = bitselect(s[10]^s[12],s[10],s[11]); s[11] = bitselect(s[11]^s[13],s[11],s[12]); s[12] = bitselect(s[12]^s[14],s[12],s[13]); s[13] = bitselect(s[13]^u,s[13],s[14]); s[14] = bitselect(s[14]^v,s[14],u);
		u = s[15]; v = s[16]; s[15] = bitselect(s[15]^s[17],s[15],s[16]); s[16] = bitselect(s[16]^s[18],s[16],s[17]); s[17] = bitselect(s[17]^s[19],s[17],s[18]); s[18] = bitselect(s[18]^u,s[18],s[19]); s[19] = bitselect(s[19]^v,s[19],u);
		u = s[20]; v = s[21]; s[20] = bitselect(s[20]^s[22],s[20],s[21]); s[21] = bitselect(s[21]^s[23],s[21],s[22]); s[22] = bitselect(s[22]^s[24],s[22],s[23]); s[23] = bitselect(s[23]^u,s[23],s[24]); s[24] = bitselect(s[24]^v,s[24],u);


		s[0] ^= keccak_round_constants35[i]; }
	}
}

//
// END KECCAK32
//


//
// BEGIN SKEIN256
//

__constant static const ulong SKEIN_IV512_256[8] = {
	0xCCD044A12FDB3E13UL, 0xE83590301A79A9EBUL,
	0x55AEA0614F816E6FUL, 0x2A2767A4AE9B94DBUL,
	0xEC06025E74DD7683UL, 0xE7A436CDC4746251UL,
	0xC36FBAF9393AD185UL, 0x3EEDBA1833EDFC13UL
};

__constant static const ulong SKEIN_IV256[4] = {
	0xFC9DA860D048B449UL, 0x2FCA66479FA7D833UL,
	0xB33BC3896656840FUL, 0x6A54E920FDE8DA69UL
};

__constant static const int ROT256[8][4] = {
	{ 46, 36, 19, 37 },
	{ 33, 27, 14, 42 },
	{ 17, 49, 36, 39 },
	{ 44, 9, 54, 56  },
	{ 39, 30, 34, 24 },
	{ 13, 50, 10, 17 },
	{ 25, 29, 39, 43 },
	{ 8, 35, 56, 22  }
};

__constant static const ulong skein_ks_parity = 0x1BD11BDAA9FC1A22;

__constant static const ulong skein_t12[6] =
{ 0x20UL,
0xf000000000000000UL,
0xf000000000000020UL,
0x08UL,
0xff00000000000000UL,
0xff00000000000008UL
};


#define Round512(p0,p1,p2,p3,p4,p5,p6,p7,ROT)  { \
p0 += p1; p1 = SPH_ROTL64(p1, ROT256[ROT][0]);  p1 ^= p0; \
p2 += p3; p3 = SPH_ROTL64(p3, ROT256[ROT][1]);  p3 ^= p2; \
p4 += p5; p5 = SPH_ROTL64(p5, ROT256[ROT][2]);  p5 ^= p4; \
p6 += p7; p7 = SPH_ROTL64(p7, ROT256[ROT][3]);  p7 ^= p6; \
} 

#define Round_8_512(p0, p1, p2, p3, p4, p5, p6, p7, R) { \
	    Round512(p0, p1, p2, p3, p4, p5, p6, p7, 0); \
	    Round512(p2, p1, p4, p7, p6, p5, p0, p3, 1); \
	    Round512(p4, p1, p6, p3, p0, p5, p2, p7, 2); \
	    Round512(p6, p1, p0, p7, p2, p5, p4, p3, 3); \
	    p0 += h[((R)+0) % 9]; \
      p1 += h[((R)+1) % 9]; \
      p2 += h[((R)+2) % 9]; \
      p3 += h[((R)+3) % 9]; \
      p4 += h[((R)+4) % 9]; \
      p5 += h[((R)+5) % 9] + t[((R)+0) % 3]; \
      p6 += h[((R)+6) % 9] + t[((R)+1) % 3]; \
      p7 += h[((R)+7) % 9] + R; \
		Round512(p0, p1, p2, p3, p4, p5, p6, p7, 4); \
		Round512(p2, p1, p4, p7, p6, p5, p0, p3, 5); \
		Round512(p4, p1, p6, p3, p0, p5, p2, p7, 6); \
		Round512(p6, p1, p0, p7, p2, p5, p4, p3, 7); \
		p0 += h[((R)+1) % 9]; \
		p1 += h[((R)+2) % 9]; \
		p2 += h[((R)+3) % 9]; \
		p3 += h[((R)+4) % 9]; \
		p4 += h[((R)+5) % 9]; \
		p5 += h[((R)+6) % 9] + t[((R)+1) % 3]; \
		p6 += h[((R)+7) % 9] + t[((R)+2) % 3]; \
		p7 += h[((R)+8) % 9] + (R+1); \
}


//
// END SKEIN256
//

//
// BEGIN BMW
//


#define shl(x, n)            ((x) << (n))
#define shr(x, n)            ((x) >> (n))
//#define SHR(x, n) SHR2(x, n) 
//#define SHL(x, n) SHL2(x, n) 


#define ss0(x)  (shr((x), 1) ^ shl((x), 3) ^ SPH_ROTL32((x),  4) ^ SPH_ROTL32((x), 19))
#define ss1(x)  (shr((x), 1) ^ shl((x), 2) ^ SPH_ROTL32((x),  8) ^ SPH_ROTL32((x), 23))
#define ss2(x)  (shr((x), 2) ^ shl((x), 1) ^ SPH_ROTL32((x), 12) ^ SPH_ROTL32((x), 25))
#define ss3(x)  (shr((x), 2) ^ shl((x), 2) ^ SPH_ROTL32((x), 15) ^ SPH_ROTL32((x), 29))
#define ss4(x)  (shr((x), 1) ^ (x))
#define ss5(x)  (shr((x), 2) ^ (x))
#define rs1(x) SPH_ROTL32((x),  3)
#define rs2(x) SPH_ROTL32((x),  7)
#define rs3(x) SPH_ROTL32((x), 13)
#define rs4(x) SPH_ROTL32((x), 16)
#define rs5(x) SPH_ROTL32((x), 19)
#define rs6(x) SPH_ROTL32((x), 23)
#define rs7(x) SPH_ROTL32((x), 27)

/* Message expansion function 1 */
uint expand32_1(int i, const uint *M32, const uint *H, const uint *Q)
{

	return (ss1(Q[i - 16]) + ss2(Q[i - 15]) + ss3(Q[i - 14]) + ss0(Q[i - 13])
		+ ss1(Q[i - 12]) + ss2(Q[i - 11]) + ss3(Q[i - 10]) + ss0(Q[i - 9])
		+ ss1(Q[i - 8]) + ss2(Q[i - 7]) + ss3(Q[i - 6]) + ss0(Q[i - 5])
		+ ss1(Q[i - 4]) + ss2(Q[i - 3]) + ss3(Q[i - 2]) + ss0(Q[i - 1])
		+ ((i*(0x05555555ul) + SPH_ROTL32(M32[(i - 16) % 16], ((i - 16) % 16) + 1) + SPH_ROTL32(M32[(i - 13) % 16], ((i - 13) % 16) + 1) - SPH_ROTL32(M32[(i - 6) % 16], ((i - 6) % 16) + 1)) ^ H[(i - 16 + 7) % 16]));

}

/* Message expansion function 2 */
uint expand32_2(int i, const uint *M32, const uint *H, const uint *Q)
{

	return (Q[i - 16] + rs1(Q[i - 15]) + Q[i - 14] + rs2(Q[i - 13])
		+ Q[i - 12] + rs3(Q[i - 11]) + Q[i - 10] + rs4(Q[i - 9])
		+ Q[i - 8] + rs5(Q[i - 7]) + Q[i - 6] + rs6(Q[i - 5])
		+ Q[i - 4] + rs7(Q[i - 3]) + ss4(Q[i - 2]) + ss5(Q[i - 1])
		+ ((i*(0x05555555ul) + SPH_ROTL32(M32[(i - 16) % 16], ((i - 16) % 16) + 1) + SPH_ROTL32(M32[(i - 13) % 16], ((i - 13) % 16) + 1) - SPH_ROTL32(M32[(i - 6) % 16], ((i - 6) % 16) + 1)) ^ H[(i - 16 + 7) % 16]));

}

void Compression256(const uint *M32, uint *H)
{
	uint XL32, XH32, Q[32];


	Q[0] = (M32[5] ^ H[5]) - (M32[7] ^ H[7]) + (M32[10] ^ H[10]) + (M32[13] ^ H[13]) + (M32[14] ^ H[14]);
	Q[1] = (M32[6] ^ H[6]) - (M32[8] ^ H[8]) + (M32[11] ^ H[11]) + (M32[14] ^ H[14]) - (M32[15] ^ H[15]);
	Q[2] = (M32[0] ^ H[0]) + (M32[7] ^ H[7]) + (M32[9] ^ H[9]) - (M32[12] ^ H[12]) + (M32[15] ^ H[15]);
	Q[3] = (M32[0] ^ H[0]) - (M32[1] ^ H[1]) + (M32[8] ^ H[8]) - (M32[10] ^ H[10]) + (M32[13] ^ H[13]);
	Q[4] = (M32[1] ^ H[1]) + (M32[2] ^ H[2]) + (M32[9] ^ H[9]) - (M32[11] ^ H[11]) - (M32[14] ^ H[14]);
	Q[5] = (M32[3] ^ H[3]) - (M32[2] ^ H[2]) + (M32[10] ^ H[10]) - (M32[12] ^ H[12]) + (M32[15] ^ H[15]);
	Q[6] = (M32[4] ^ H[4]) - (M32[0] ^ H[0]) - (M32[3] ^ H[3]) - (M32[11] ^ H[11]) + (M32[13] ^ H[13]);
	Q[7] = (M32[1] ^ H[1]) - (M32[4] ^ H[4]) - (M32[5] ^ H[5]) - (M32[12] ^ H[12]) - (M32[14] ^ H[14]);
	Q[8] = (M32[2] ^ H[2]) - (M32[5] ^ H[5]) - (M32[6] ^ H[6]) + (M32[13] ^ H[13]) - (M32[15] ^ H[15]);
	Q[9] = (M32[0] ^ H[0]) - (M32[3] ^ H[3]) + (M32[6] ^ H[6]) - (M32[7] ^ H[7]) + (M32[14] ^ H[14]);
	Q[10] = (M32[8] ^ H[8]) - (M32[1] ^ H[1]) - (M32[4] ^ H[4]) - (M32[7] ^ H[7]) + (M32[15] ^ H[15]);
	Q[11] = (M32[8] ^ H[8]) - (M32[0] ^ H[0]) - (M32[2] ^ H[2]) - (M32[5] ^ H[5]) + (M32[9] ^ H[9]);
	Q[12] = (M32[1] ^ H[1]) + (M32[3] ^ H[3]) - (M32[6] ^ H[6]) - (M32[9] ^ H[9]) + (M32[10] ^ H[10]);
	Q[13] = (M32[2] ^ H[2]) + (M32[4] ^ H[4]) + (M32[7] ^ H[7]) + (M32[10] ^ H[10]) + (M32[11] ^ H[11]);
	Q[14] = (M32[3] ^ H[3]) - (M32[5] ^ H[5]) + (M32[8] ^ H[8]) - (M32[11] ^ H[11]) - (M32[12] ^ H[12]);
	Q[15] = (M32[12] ^ H[12]) - (M32[4] ^ H[4]) - (M32[6] ^ H[6]) - (M32[9] ^ H[9]) + (M32[13] ^ H[13]);

	/*  Diffuse the differences in every word in a bijective manner with ssi, and then add the values of the previous double pipe.*/
	Q[0] = ss0(Q[0]) + H[1];
	Q[1] = ss1(Q[1]) + H[2];
	Q[2] = ss2(Q[2]) + H[3];
	Q[3] = ss3(Q[3]) + H[4];
	Q[4] = ss4(Q[4]) + H[5];
	Q[5] = ss0(Q[5]) + H[6];
	Q[6] = ss1(Q[6]) + H[7];
	Q[7] = ss2(Q[7]) + H[8];
	Q[8] = ss3(Q[8]) + H[9];
	Q[9] = ss4(Q[9]) + H[10];
	Q[10] = ss0(Q[10]) + H[11];
	Q[11] = ss1(Q[11]) + H[12];
	Q[12] = ss2(Q[12]) + H[13];
	Q[13] = ss3(Q[13]) + H[14];
	Q[14] = ss4(Q[14]) + H[15];
	Q[15] = ss0(Q[15]) + H[0];

	/* This is the Message expansion or f_1 in the documentation.       */
	/* It has 16 rounds.                                                */
	/* Blue Midnight Wish has two tunable security parameters.          */
	/* The parameters are named EXPAND_1_ROUNDS and EXPAND_2_ROUNDS.    */
	/* The following relation for these parameters should is satisfied: */
	/* EXPAND_1_ROUNDS + EXPAND_2_ROUNDS = 16                           */
#pragma unroll
	for (int i = 0; i<2; i++)
		Q[i + 16] = expand32_1(i + 16, M32, H, Q);

#pragma unroll
	for (int i = 2; i<16; i++)
		Q[i + 16] = expand32_2(i + 16, M32, H, Q);

	/* Blue Midnight Wish has two temporary cummulative variables that accumulate via XORing */
	/* 16 new variables that are prooduced in the Message Expansion part.                    */
	XL32 = Q[16] ^ Q[17] ^ Q[18] ^ Q[19] ^ Q[20] ^ Q[21] ^ Q[22] ^ Q[23];
	XH32 = XL32^Q[24] ^ Q[25] ^ Q[26] ^ Q[27] ^ Q[28] ^ Q[29] ^ Q[30] ^ Q[31];


	/*  This part is the function f_2 - in the documentation            */

	/*  Compute the double chaining pipe for the next message block.    */
	H[0] = (shl(XH32, 5) ^ shr(Q[16], 5) ^ M32[0]) + (XL32    ^ Q[24] ^ Q[0]);
	H[1] = (shr(XH32, 7) ^ shl(Q[17], 8) ^ M32[1]) + (XL32    ^ Q[25] ^ Q[1]);
	H[2] = (shr(XH32, 5) ^ shl(Q[18], 5) ^ M32[2]) + (XL32    ^ Q[26] ^ Q[2]);
	H[3] = (shr(XH32, 1) ^ shl(Q[19], 5) ^ M32[3]) + (XL32    ^ Q[27] ^ Q[3]);
	H[4] = (shr(XH32, 3) ^ Q[20] ^ M32[4]) + (XL32    ^ Q[28] ^ Q[4]);
	H[5] = (shl(XH32, 6) ^ shr(Q[21], 6) ^ M32[5]) + (XL32    ^ Q[29] ^ Q[5]);
	H[6] = (shr(XH32, 4) ^ shl(Q[22], 6) ^ M32[6]) + (XL32    ^ Q[30] ^ Q[6]);
	H[7] = (shr(XH32, 11) ^ shl(Q[23], 2) ^ M32[7]) + (XL32    ^ Q[31] ^ Q[7]);

	H[8] = SPH_ROTL32(H[4], 9) + (XH32     ^     Q[24] ^ M32[8]) + (shl(XL32, 8) ^ Q[23] ^ Q[8]);
	H[9] = SPH_ROTL32(H[5], 10) + (XH32     ^     Q[25] ^ M32[9]) + (shr(XL32, 6) ^ Q[16] ^ Q[9]);
	H[10] = SPH_ROTL32(H[6], 11) + (XH32     ^     Q[26] ^ M32[10]) + (shl(XL32, 6) ^ Q[17] ^ Q[10]);
	H[11] = SPH_ROTL32(H[7], 12) + (XH32     ^     Q[27] ^ M32[11]) + (shl(XL32, 4) ^ Q[18] ^ Q[11]);
	H[12] = SPH_ROTL32(H[0], 13) + (XH32     ^     Q[28] ^ M32[12]) + (shr(XL32, 3) ^ Q[19] ^ Q[12]);
	H[13] = SPH_ROTL32(H[1], 14) + (XH32     ^     Q[29] ^ M32[13]) + (shr(XL32, 4) ^ Q[20] ^ Q[13]);
	H[14] = SPH_ROTL32(H[2], 15) + (XH32     ^     Q[30] ^ M32[14]) + (shr(XL32, 7) ^ Q[21] ^ Q[14]);
	H[15] = SPH_ROTL32(H[3], 16) + (XH32     ^     Q[31] ^ M32[15]) + (shr(XL32, 2) ^ Q[22] ^ Q[15]);
}

ulong Compression256_last(const uint *M32, const uint *H)
{
	uint XL32, XH32, Q[32];


	Q[0] = (M32[5] ^ H[5]) - (M32[7] ^ H[7]) + (M32[10] ^ H[10]) + (M32[13] ^ H[13]) + (M32[14] ^ H[14]);
	Q[1] = (M32[6] ^ H[6]) - (M32[8] ^ H[8]) + (M32[11] ^ H[11]) + (M32[14] ^ H[14]) - (M32[15] ^ H[15]);
	Q[2] = (M32[0] ^ H[0]) + (M32[7] ^ H[7]) + (M32[9] ^ H[9]) - (M32[12] ^ H[12]) + (M32[15] ^ H[15]);
	Q[3] = (M32[0] ^ H[0]) - (M32[1] ^ H[1]) + (M32[8] ^ H[8]) - (M32[10] ^ H[10]) + (M32[13] ^ H[13]);
	Q[4] = (M32[1] ^ H[1]) + (M32[2] ^ H[2]) + (M32[9] ^ H[9]) - (M32[11] ^ H[11]) - (M32[14] ^ H[14]);
	Q[5] = (M32[3] ^ H[3]) - (M32[2] ^ H[2]) + (M32[10] ^ H[10]) - (M32[12] ^ H[12]) + (M32[15] ^ H[15]);
	Q[6] = (M32[4] ^ H[4]) - (M32[0] ^ H[0]) - (M32[3] ^ H[3]) - (M32[11] ^ H[11]) + (M32[13] ^ H[13]);
	Q[7] = (M32[1] ^ H[1]) - (M32[4] ^ H[4]) - (M32[5] ^ H[5]) - (M32[12] ^ H[12]) - (M32[14] ^ H[14]);
	Q[8] = (M32[2] ^ H[2]) - (M32[5] ^ H[5]) - (M32[6] ^ H[6]) + (M32[13] ^ H[13]) - (M32[15] ^ H[15]);
	Q[9] = (M32[0] ^ H[0]) - (M32[3] ^ H[3]) + (M32[6] ^ H[6]) - (M32[7] ^ H[7]) + (M32[14] ^ H[14]);
	Q[10] = (M32[8] ^ H[8]) - (M32[1] ^ H[1]) - (M32[4] ^ H[4]) - (M32[7] ^ H[7]) + (M32[15] ^ H[15]);
	Q[11] = (M32[8] ^ H[8]) - (M32[0] ^ H[0]) - (M32[2] ^ H[2]) - (M32[5] ^ H[5]) + (M32[9] ^ H[9]);
	Q[12] = (M32[1] ^ H[1]) + (M32[3] ^ H[3]) - (M32[6] ^ H[6]) - (M32[9] ^ H[9]) + (M32[10] ^ H[10]);
	Q[13] = (M32[2] ^ H[2]) + (M32[4] ^ H[4]) + (M32[7] ^ H[7]) + (M32[10] ^ H[10]) + (M32[11] ^ H[11]);
	Q[14] = (M32[3] ^ H[3]) - (M32[5] ^ H[5]) + (M32[8] ^ H[8]) - (M32[11] ^ H[11]) - (M32[12] ^ H[12]);
	Q[15] = (M32[12] ^ H[12]) - (M32[4] ^ H[4]) - (M32[6] ^ H[6]) - (M32[9] ^ H[9]) + (M32[13] ^ H[13]);

	/*  Diffuse the differences in every word in a bijective manner with ssi, and then add the values of the previous double pipe.*/
	Q[0] = ss0(Q[0]) + H[1];
	Q[1] = ss1(Q[1]) + H[2];
	Q[2] = ss2(Q[2]) + H[3];
	Q[3] = ss3(Q[3]) + H[4];
	Q[4] = ss4(Q[4]) + H[5];
	Q[5] = ss0(Q[5]) + H[6];
	Q[6] = ss1(Q[6]) + H[7];
	Q[7] = ss2(Q[7]) + H[8];
	Q[8] = ss3(Q[8]) + H[9];
	Q[9] = ss4(Q[9]) + H[10];
	Q[10] = ss0(Q[10]) + H[11];
	Q[11] = ss1(Q[11]) + H[12];
	Q[12] = ss2(Q[12]) + H[13];
	Q[13] = ss3(Q[13]) + H[14];
	Q[14] = ss4(Q[14]) + H[15];
	Q[15] = ss0(Q[15]) + H[0];

	/* This is the Message expansion or f_1 in the documentation.       */
	/* It has 16 rounds.                                                */
	/* Blue Midnight Wish has two tunable security parameters.          */
	/* The parameters are named EXPAND_1_ROUNDS and EXPAND_2_ROUNDS.    */
	/* The following relation for these parameters should is satisfied: */
	/* EXPAND_1_ROUNDS + EXPAND_2_ROUNDS = 16                           */
#pragma unroll
	for (int i = 0; i<2; i++)
		Q[i + 16] = expand32_1(i + 16, M32, H, Q);

#pragma unroll
	for (int i = 2; i<16; i++)
		Q[i + 16] = expand32_2(i + 16, M32, H, Q);

	/* Blue Midnight Wish has two temporary cummulative variables that accumulate via XORing */
	/* 16 new variables that are prooduced in the Message Expansion part.                    */
	XL32 = Q[16] ^ Q[17] ^ Q[18] ^ Q[19] ^ Q[20] ^ Q[21] ^ Q[22] ^ Q[23];
	XH32 = XL32^Q[24] ^ Q[25] ^ Q[26] ^ Q[27] ^ Q[28] ^ Q[29] ^ Q[30] ^ Q[31];

	/*  This part is the function f_2 - in the documentation            */

	/*  Compute the double chaining pipe for the next message block.    */

	uint ret[2];
	uint H2 = (shr(XH32, 5) ^ shl(Q[18], 5) ^ M32[2]) + (XL32    ^ Q[26] ^ Q[2]);
	uint H3 = (shr(XH32, 1) ^ shl(Q[19], 5) ^ M32[3]) + (XL32    ^ Q[27] ^ Q[3]);
	ret[0] = SPH_ROTL32(H2, 15) + (XH32     ^     Q[30] ^ M32[14]) + (shr(XL32, 7) ^ Q[21] ^ Q[14]);
	ret[1] = SPH_ROTL32(H3, 16) + (XH32     ^     Q[31] ^ M32[15]) + (shr(XL32, 2) ^ Q[22] ^ Q[15]);

	return *((ulong*)ret);
}




//
// END BMW
//


//
// BEGIN CUBEHASH
//

#if !defined SPH_CUBEHASH_UNROLL
#define SPH_CUBEHASH_UNROLL   0
#endif

__constant static const uint CUBEHASH_IV512[] = {
  SPH_C32(0x2AEA2A61), SPH_C32(0x50F494D4), SPH_C32(0x2D538B8B),
  SPH_C32(0x4167D83E), SPH_C32(0x3FEE2313), SPH_C32(0xC701CF8C),
  SPH_C32(0xCC39968E), SPH_C32(0x50AC5695), SPH_C32(0x4D42C787),
  SPH_C32(0xA647A8B3), SPH_C32(0x97CF0BEF), SPH_C32(0x825B4537),
  SPH_C32(0xEEF864D2), SPH_C32(0xF22090C4), SPH_C32(0xD0E5CD33),
  SPH_C32(0xA23911AE), SPH_C32(0xFCD398D9), SPH_C32(0x148FE485),
  SPH_C32(0x1B017BEF), SPH_C32(0xB6444532), SPH_C32(0x6A536159),
  SPH_C32(0x2FF5781C), SPH_C32(0x91FA7934), SPH_C32(0x0DBADEA9),
  SPH_C32(0xD65C8A2B), SPH_C32(0xA5A70E75), SPH_C32(0xB1C62456),
  SPH_C32(0xBC796576), SPH_C32(0x1921C8F7), SPH_C32(0xE7989AF1),
  SPH_C32(0x7795D246), SPH_C32(0xD43E3B44)
};

#define T32      SPH_T32
#define ROTL32   SPH_ROTL32

#define ROUND_EVEN   do { \
    x[0x10] = T32(x[0x0] + x[0x10]); \
    x[0x0] = ROTL32(x[0x0], 7); \
    x[0x11] = T32(x[0x1] + x[0x11]); \
    x[0x1] = ROTL32(x[0x1], 7); \
    x[0x12] = T32(x[0x2] + x[0x12]); \
    x[0x2] = ROTL32(x[0x2], 7); \
    x[0x13] = T32(x[0x3] + x[0x13]); \
    x[0x3] = ROTL32(x[0x3], 7); \
    x[0x14] = T32(x[0x4] + x[0x14]); \
    x[0x4] = ROTL32(x[0x4], 7); \
    x[0x15] = T32(x[0x5] + x[0x15]); \
    x[0x5] = ROTL32(x[0x5], 7); \
    x[0x16] = T32(x[0x6] + x[0x16]); \
    x[0x6] = ROTL32(x[0x6], 7); \
    x[0x17] = T32(x[0x7] + x[0x17]); \
    x[0x7] = ROTL32(x[0x7], 7); \
    x[0x18] = T32(x[0x8] + x[0x18]); \
    x[0x8] = ROTL32(x[0x8], 7); \
    x[0x19] = T32(x[0x9] + x[0x19]); \
    x[0x9] = ROTL32(x[0x9], 7); \
    x[0x1a] = T32(x[0xa] + x[0x1a]); \
    x[0xa] = ROTL32(x[0xa], 7); \
    x[0x1b] = T32(x[0xb] + x[0x1b]); \
    x[0xb] = ROTL32(x[0xb], 7); \
    x[0x1c] = T32(x[0xc] + x[0x1c]); \
    x[0xc] = ROTL32(x[0xc], 7); \
    x[0x1d] = T32(x[0xd] + x[0x1d]); \
    x[0xd] = ROTL32(x[0xd], 7); \
    x[0x1e] = T32(x[0xe] + x[0x1e]); \
    x[0xe] = ROTL32(x[0xe], 7); \
    x[0x1f] = T32(x[0xf] + x[0x1f]); \
    x[0xf] = ROTL32(x[0xf], 7); \
    x[0x8] ^= x[0x10]; \
    x[0x9] ^= x[0x11]; \
    x[0xa] ^= x[0x12]; \
    x[0xb] ^= x[0x13]; \
    x[0xc] ^= x[0x14]; \
    x[0xd] ^= x[0x15]; \
    x[0xe] ^= x[0x16]; \
    x[0xf] ^= x[0x17]; \
    x[0x0] ^= x[0x18]; \
    x[0x1] ^= x[0x19]; \
    x[0x2] ^= x[0x1a]; \
    x[0x3] ^= x[0x1b]; \
    x[0x4] ^= x[0x1c]; \
    x[0x5] ^= x[0x1d]; \
    x[0x6] ^= x[0x1e]; \
    x[0x7] ^= x[0x1f]; \
    x[0x12] = T32(x[0x8] + x[0x12]); \
    x[0x8] = ROTL32(x[0x8], 11); \
    x[0x13] = T32(x[0x9] + x[0x13]); \
    x[0x9] = ROTL32(x[0x9], 11); \
    x[0x10] = T32(x[0xa] + x[0x10]); \
    x[0xa] = ROTL32(x[0xa], 11); \
    x[0x11] = T32(x[0xb] + x[0x11]); \
    x[0xb] = ROTL32(x[0xb], 11); \
    x[0x16] = T32(x[0xc] + x[0x16]); \
    x[0xc] = ROTL32(x[0xc], 11); \
    x[0x17] = T32(x[0xd] + x[0x17]); \
    x[0xd] = ROTL32(x[0xd], 11); \
    x[0x14] = T32(x[0xe] + x[0x14]); \
    x[0xe] = ROTL32(x[0xe], 11); \
    x[0x15] = T32(x[0xf] + x[0x15]); \
    x[0xf] = ROTL32(x[0xf], 11); \
    x[0x1a] = T32(x[0x0] + x[0x1a]); \
    x[0x0] = ROTL32(x[0x0], 11); \
    x[0x1b] = T32(x[0x1] + x[0x1b]); \
    x[0x1] = ROTL32(x[0x1], 11); \
    x[0x18] = T32(x[0x2] + x[0x18]); \
    x[0x2] = ROTL32(x[0x2], 11); \
    x[0x19] = T32(x[0x3] + x[0x19]); \
    x[0x3] = ROTL32(x[0x3], 11); \
    x[0x1e] = T32(x[0x4] + x[0x1e]); \
    x[0x4] = ROTL32(x[0x4], 11); \
    x[0x1f] = T32(x[0x5] + x[0x1f]); \
    x[0x5] = ROTL32(x[0x5], 11); \
    x[0x1c] = T32(x[0x6] + x[0x1c]); \
    x[0x6] = ROTL32(x[0x6], 11); \
    x[0x1d] = T32(x[0x7] + x[0x1d]); \
    x[0x7] = ROTL32(x[0x7], 11); \
    x[0xc] ^= x[0x12]; \
    x[0xd] ^= x[0x13]; \
    x[0xe] ^= x[0x10]; \
    x[0xf] ^= x[0x11]; \
    x[0x8] ^= x[0x16]; \
    x[0x9] ^= x[0x17]; \
    x[0xa] ^= x[0x14]; \
    x[0xb] ^= x[0x15]; \
    x[0x4] ^= x[0x1a]; \
    x[0x5] ^= x[0x1b]; \
    x[0x6] ^= x[0x18]; \
    x[0x7] ^= x[0x19]; \
    x[0x0] ^= x[0x1e]; \
    x[0x1] ^= x[0x1f]; \
    x[0x2] ^= x[0x1c]; \
    x[0x3] ^= x[0x1d]; \
  } while (0)

#define ROUND_ODD   do { \
    x[0x13] = T32(x[0xc] + x[0x13]); \
    x[0xc] = ROTL32(x[0xc], 7); \
    x[0x12] = T32(x[0xd] + x[0x12]); \
    x[0xd] = ROTL32(x[0xd], 7); \
    x[0x11] = T32(x[0xe] + x[0x11]); \
    x[0xe] = ROTL32(x[0xe], 7); \
    x[0x10] = T32(x[0xf] + x[0x10]); \
    x[0xf] = ROTL32(x[0xf], 7); \
    x[0x17] = T32(x[0x8] + x[0x17]); \
    x[0x8] = ROTL32(x[0x8], 7); \
    x[0x16] = T32(x[0x9] + x[0x16]); \
    x[0x9] = ROTL32(x[0x9], 7); \
    x[0x15] = T32(x[0xa] + x[0x15]); \
    x[0xa] = ROTL32(x[0xa], 7); \
    x[0x14] = T32(x[0xb] + x[0x14]); \
    x[0xb] = ROTL32(x[0xb], 7); \
    x[0x1b] = T32(x[0x4] + x[0x1b]); \
    x[0x4] = ROTL32(x[0x4], 7); \
    x[0x1a] = T32(x[0x5] + x[0x1a]); \
    x[0x5] = ROTL32(x[0x5], 7); \
    x[0x19] = T32(x[0x6] + x[0x19]); \
    x[0x6] = ROTL32(x[0x6], 7); \
    x[0x18] = T32(x[0x7] + x[0x18]); \
    x[0x7] = ROTL32(x[0x7], 7); \
    x[0x1f] = T32(x[0x0] + x[0x1f]); \
    x[0x0] = ROTL32(x[0x0], 7); \
    x[0x1e] = T32(x[0x1] + x[0x1e]); \
    x[0x1] = ROTL32(x[0x1], 7); \
    x[0x1d] = T32(x[0x2] + x[0x1d]); \
    x[0x2] = ROTL32(x[0x2], 7); \
    x[0x1c] = T32(x[0x3] + x[0x1c]); \
    x[0x3] = ROTL32(x[0x3], 7); \
    x[0x4] ^= x[0x13]; \
    x[0x5] ^= x[0x12]; \
    x[0x6] ^= x[0x11]; \
    x[0x7] ^= x[0x10]; \
    x[0x0] ^= x[0x17]; \
    x[0x1] ^= x[0x16]; \
    x[0x2] ^= x[0x15]; \
    x[0x3] ^= x[0x14]; \
    x[0xc] ^= x[0x1b]; \
    x[0xd] ^= x[0x1a]; \
    x[0xe] ^= x[0x19]; \
    x[0xf] ^= x[0x18]; \
    x[0x8] ^= x[0x1f]; \
    x[0x9] ^= x[0x1e]; \
    x[0xa] ^= x[0x1d]; \
    x[0xb] ^= x[0x1c]; \
    x[0x11] = T32(x[0x4] + x[0x11]); \
    x[0x4] = ROTL32(x[0x4], 11); \
    x[0x10] = T32(x[0x5] + x[0x10]); \
    x[0x5] = ROTL32(x[0x5], 11); \
    x[0x13] = T32(x[0x6] + x[0x13]); \
    x[0x6] = ROTL32(x[0x6], 11); \
    x[0x12] = T32(x[0x7] + x[0x12]); \
    x[0x7] = ROTL32(x[0x7], 11); \
    x[0x15] = T32(x[0x0] + x[0x15]); \
    x[0x0] = ROTL32(x[0x0], 11); \
    x[0x14] = T32(x[0x1] + x[0x14]); \
    x[0x1] = ROTL32(x[0x1], 11); \
    x[0x17] = T32(x[0x2] + x[0x17]); \
    x[0x2] = ROTL32(x[0x2], 11); \
    x[0x16] = T32(x[0x3] + x[0x16]); \
    x[0x3] = ROTL32(x[0x3], 11); \
    x[0x19] = T32(x[0xc] + x[0x19]); \
    x[0xc] = ROTL32(x[0xc], 11); \
    x[0x18] = T32(x[0xd] + x[0x18]); \
    x[0xd] = ROTL32(x[0xd], 11); \
    x[0x1b] = T32(x[0xe] + x[0x1b]); \
    x[0xe] = ROTL32(x[0xe], 11); \
    x[0x1a] = T32(x[0xf] + x[0x1a]); \
    x[0xf] = ROTL32(x[0xf], 11); \
    x[0x1d] = T32(x[0x8] + x[0x1d]); \
    x[0x8] = ROTL32(x[0x8], 11); \
    x[0x1c] = T32(x[0x9] + x[0x1c]); \
    x[0x9] = ROTL32(x[0x9], 11); \
    x[0x1f] = T32(x[0xa] + x[0x1f]); \
    x[0xa] = ROTL32(x[0xa], 11); \
    x[0x1e] = T32(x[0xb] + x[0x1e]); \
    x[0xb] = ROTL32(x[0xb], 11); \
    x[0x0] ^= x[0x11]; \
    x[0x1] ^= x[0x10]; \
    x[0x2] ^= x[0x13]; \
    x[0x3] ^= x[0x12]; \
    x[0x4] ^= x[0x15]; \
    x[0x5] ^= x[0x14]; \
    x[0x6] ^= x[0x17]; \
    x[0x7] ^= x[0x16]; \
    x[0x8] ^= x[0x19]; \
    x[0x9] ^= x[0x18]; \
    x[0xa] ^= x[0x1b]; \
    x[0xb] ^= x[0x1a]; \
    x[0xc] ^= x[0x1d]; \
    x[0xd] ^= x[0x1c]; \
    x[0xe] ^= x[0x1f]; \
    x[0xf] ^= x[0x1e]; \
  } while (0)

/*
 * There is no need to unroll all 16 rounds. The word-swapping permutation
 * is an involution, so we need to unroll an even number of rounds. On
 * "big" systems, unrolling 4 rounds yields about 97% of the speed
 * achieved with full unrolling; and it keeps the code more compact
 * for small architectures.
 */

#if SPH_CUBEHASH_UNROLL == 2

#define SIXTEEN_ROUNDS   do { \
    for (int j = 0; j < 8; j ++) { \
      ROUND_EVEN; \
      ROUND_ODD; \
    } \
  } while (0)

#elif SPH_CUBEHASH_UNROLL == 4

#define SIXTEEN_ROUNDS   do { \
    for (int j = 0; j < 4; j ++) { \
      ROUND_EVEN; \
      ROUND_ODD; \
      ROUND_EVEN; \
      ROUND_ODD; \
    } \
  } while (0)

#elif SPH_CUBEHASH_UNROLL == 8

#define SIXTEEN_ROUNDS   do { \
    for (int j = 0; j < 2; j ++) { \
      ROUND_EVEN; \
      ROUND_ODD; \
      ROUND_EVEN; \
      ROUND_ODD; \
      ROUND_EVEN; \
      ROUND_ODD; \
      ROUND_EVEN; \
      ROUND_ODD; \
    } \
  } while (0)

#else

#define SIXTEEN_ROUNDS   do { \
    ROUND_EVEN; \
    ROUND_ODD; \
    ROUND_EVEN; \
    ROUND_ODD; \
    ROUND_EVEN; \
    ROUND_ODD; \
    ROUND_EVEN; \
    ROUND_ODD; \
    ROUND_EVEN; \
    ROUND_ODD; \
    ROUND_EVEN; \
    ROUND_ODD; \
    ROUND_EVEN; \
    ROUND_ODD; \
    ROUND_EVEN; \
    ROUND_ODD; \
  } while (0)

#endif

//
// END CUBEHASH
//

//
// BEGIN LYRA
//

// wubwub

// LYRA2 PREPROCESSOR MACROS


// replicate build env
#define LYRA_SCRATCHBUF_SIZE 1536
#define LYRA_SCRATCHBUF_SIZE_ULONG4 (LYRA_SCRATCHBUF_SIZE / (32))
#define memshift 3

// opencl versions
#define ROTL64(x,n) rotate(x,(ulong)n)
#define ROTR64(x,n) rotate(x,(ulong)(64-n))
#define SWAP32(x) as_ulong(as_uint2(x).s10)
//#define ROTL64(x,n) SPH_ROTL64(x, n)
//#define ROTR64(x,n) SPH_ROTR64(x, n)
//#define SWAP32(x) sph_bswap32(x)


#define LYRA_SCOPE __global


/*One Round of the Blake2b's compression function*/

#define G_old(a,b,c,d) \
  do { \
	a += b; d ^= a; d = ROTR64(d, 32); \
	c += d; b ^= c; b = ROTR64(b, 24); \
	a += b; d ^= a; d = ROTR64(d, 16); \
	c += d; b ^= c; b = ROTR64(b, 63); \
\
  } while (0)

#define round_lyra(s)  \
 do { \
	 G_old(s[0].x, s[1].x, s[2].x, s[3].x); \
     G_old(s[0].y, s[1].y, s[2].y, s[3].y); \
     G_old(s[0].z, s[1].z, s[2].z, s[3].z); \
     G_old(s[0].w, s[1].w, s[2].w, s[3].w); \
     G_old(s[0].x, s[1].y, s[2].z, s[3].w); \
     G_old(s[0].y, s[1].z, s[2].w, s[3].x); \
     G_old(s[0].z, s[1].w, s[2].x, s[3].y); \
     G_old(s[0].w, s[1].x, s[2].y, s[3].z); \
 } while(0)

#define G(a,b,c,d) \
  do { \
	a += b; d ^= a; d = SWAP32(d); \
	c += d; b ^= c; b = round_lyra(b,24); \
	a += b; d ^= a; d = ROTR64(d,16); \
	c += d; b ^= c; b = ROTR64(b, 63); \
\
  } while (0)

#define SPH_ULONG4(a, b, c, d) (ulong4)(a, b, c, d)


inline void reduceDuplexf(ulong4 *state1, ulong4* state , __global ulong4* DMatrix)
{
	const uint ps1 = 0;
	const uint ps2 = (memshift * 3 + memshift * 4);
//#pragma unroll 4
	for (int i = 0; i < 4; i++)
	{
		 const uint s1 = ps1 + i*memshift;
		 const uint s2 = ps2 - i*memshift;

		 for (int j = 0; j < 3; j++)  state1[j] = (DMatrix)[j + s1];
		 for (int j = 0; j < 3; j++)  state[j] ^= state1[j];

		 round_lyra(state);

		 for (int j = 0; j < 3; j++)  state1[j] ^= state[j];
		 for (int j = 0; j < 3; j++)  (DMatrix)[j + s2] = state1[j];
	}
}

inline void reduceDuplexRowf(ulong4 *state1, ulong4* state2, uint rowIn,uint rowInOut,uint rowOut,ulong4 * state, __global ulong4 * DMatrix)
{
	const uint ps1 = (memshift * 4 * rowIn);
	const uint ps2 = (memshift * 4 * rowInOut);
	const uint ps3 = (memshift * 4 * rowOut);

	for (int i = 0; i < 4; i++)
	{
		const uint s1 = ps1 + i*memshift;
		const uint s2 = ps2 + i*memshift;
		const uint s3 = ps3 + i*memshift;

		for (int j = 0; j < 3; j++)   state1[j] = (DMatrix)[j + s1];
		for (int j = 0; j < 3; j++)   state2[j] = (DMatrix)[j + s2];
		for (int j = 0; j < 3; j++)   state1[j] += state2[j];
		for (int j = 0; j < 3; j++)   state[j] ^= state1[j];

		round_lyra(state);

		((ulong*)state2)[0] ^= ((ulong*)state)[11];
		for (int j = 0; j < 11; j++)
		{
			((ulong*)state2)[j + 1] ^= ((ulong*)state)[j];
		}

		if (rowInOut != rowOut) {
			for (int j = 0; j < 3; j++) { (DMatrix)[j + s2] = state2[j]; }
			for (int j = 0; j < 3; j++) { (DMatrix)[j + s3] ^= state[j]; }
		} else {
			for (int j = 0; j < 3; j++) { state2[j] ^= state[j]; }
			for (int j = 0; j < 3; j++) { (DMatrix)[j + s2] = state2[j]; }
		}
	}
}

inline void reduceDuplexRowSetupf(ulong4 *state1, ulong4 *state2, uint rowIn, uint rowInOut, uint rowOut, ulong4 *state,  __global ulong4* DMatrix)
{
	 uint ps1 = (memshift * 4 * rowIn);
	 uint ps2 = (memshift * 4 * rowInOut);
	 uint ps3 = (memshift * 3 + memshift * 4 * rowOut);

	 for (int i = 0; i < 4; i++)
	 {
		 uint s1 = ps1 + i*memshift;
		 uint s2 = ps2 + i*memshift;
		 uint s3 = ps3 - i*memshift;

		 for (int j = 0; j < 3; j++)  state1[j] = (DMatrix)[j + s1];

		 for (int j = 0; j < 3; j++)  state2[j] = (DMatrix)[j + s2];
		 for (int j = 0; j < 3; j++) {
			 ulong4 tmp = state1[j] + state2[j];
			 state[j] ^= tmp;
		 		 }
		 round_lyra(state);

		 for (int j = 0; j < 3; j++) {
			 state1[j] ^= state[j];
			 (DMatrix)[j + s3] = state1[j];
		 		 }

		 ((ulong*)state2)[0] ^= ((ulong*)state)[11];
		 for (int j = 0; j < 11; j++)
			 ((ulong*)state2)[j + 1] ^= ((ulong*)state)[j];
		 for (int j = 0; j < 3; j++)
			 (DMatrix)[j + s2] = state2[j];
	 }
}

// END LYRA2 PREPROCESSOR MACROS

//
// END LYRA
//


//#define DEBUG


// Hash helper functions

// blake80, in(80 bytes), out(32 bytes)
inline void blake80_noswap(__constant const uint* in_words, uint gid, __global uint* out_words)
{
	//printf("INPUT WORDS[1]: %s\n", debug_print_hash((uint*)(input_words)));
	//printf("INPUT WORDS[2]: %s\n", debug_print_hash((uint*)((uint8_t*)(input_words) + (52 - 32))));
	BLAKE256_COMPRESS32_STATE;


#ifndef PRECALC_BLAKE
	//T0 = SPH_T32(T0 + 512);
	// Input == input words, already byteswapped
	uint input_words[19];
	for (uint i=0; i<20; i++) { input_words[i] = in_words[i]; }
	BLAKE256_COMPRESS_BEGIN_DIRECT(SPH_C32(0x6a09e667), SPH_C32(0xbb67ae85), SPH_C32(0x3c6ef372), SPH_C32(0xa54ff53a),  SPH_C32(0x510e527f), SPH_C32(0x9b05688c), SPH_C32(0x1f83d9ab), SPH_C32(0x5be0cd19),
		512, 0, (input_words[0]),(input_words[1]),(input_words[2]),(input_words[3]),(input_words[4]),(input_words[5]),(input_words[6]),(input_words[7]),(input_words[8]),(input_words[9]),(input_words[10]),(input_words[11]),(input_words[12]),(input_words[13]),(input_words[14]),(input_words[15]));
#pragma unroll 
	for (uint R = 0; R< BLAKE32_ROUNDS; R++) {
		BLAKE256_GS_ALT(0, 4, 0x8, 0xC, 0x0);
		BLAKE256_GS_ALT(1, 5, 0x9, 0xD, 0x2);
		BLAKE256_GS_ALT(2, 6, 0xA, 0xE, 0x4);
		BLAKE256_GS_ALT(3, 7, 0xB, 0xF, 0x6);
		BLAKE256_GS_ALT(0, 5, 0xA, 0xF, 0x8);
		BLAKE256_GS_ALT(1, 6, 0xB, 0xC, 0xA);
		BLAKE256_GS_ALT(2, 7, 0x8, 0xD, 0xC);
		BLAKE256_GS_ALT(3, 4, 0x9, 0xE, 0xE);
	}
	BLAKE256_COMPRESS_END_DIRECT_NOSWAP(SPH_C32(0x6a09e667), SPH_C32(0xbb67ae85), SPH_C32(0x3c6ef372), SPH_C32(0xa54ff53a),  SPH_C32(0x510e527f), SPH_C32(0x9b05688c), SPH_C32(0x1f83d9ab), SPH_C32(0x5be0cd19),
		                                input_words[0], input_words[1], input_words[2], input_words[3], input_words[4], input_words[5], input_words[6], input_words[7]);
#else
	__constant const uint* input_words = in_words;
#endif

	//printf("nonce[%u] blake32 after step H=[%x,%x,%x,%x,%x,%x,%x,%x]\n", gid, input_words[0], input_words[1], input_words[2], input_words[3], input_words[4], input_words[5], input_words[6], input_words[7]);

	// blake close - filled case
	//T0 -= 512 - 128; // i.e. 128
	//T0 = SPH_T32(T0 + 512);
	//T0 = 640;

	//printf("blake32 full step T0=0x%x T1=0x%x H=[%x,%x,%x,%x,%x,%x,%x,%x] S=[%x,%x,%x,%x]\n", T0, T1, H0, H1, H2, H3, H4, H5, H6, H7, S0, S1, S2, S3);

	// NOTE: At this stage, input_words[0...7] contains the previous state of the blake hash so we can continue on similarly to before

	BLAKE256_COMPRESS_BEGIN_DIRECT(input_words[0], input_words[1], input_words[2], input_words[3], input_words[4], input_words[5], input_words[6], input_words[7], 640, 0, 
		                          (input_words[16]),(input_words[17]),(input_words[18]),(gid),2147483648,0,0,0,0,0,0,0,0,1,0,640);
#pragma unroll 
	for (uint R = 0; R< BLAKE32_ROUNDS; R++) {
		BLAKE256_GS_ALT(0, 4, 0x8, 0xC, 0x0);
		BLAKE256_GS_ALT(1, 5, 0x9, 0xD, 0x2);
		BLAKE256_GS_ALT(2, 6, 0xA, 0xE, 0x4);
		BLAKE256_GS_ALT(3, 7, 0xB, 0xF, 0x6);
		BLAKE256_GS_ALT(0, 5, 0xA, 0xF, 0x8);
		BLAKE256_GS_ALT(1, 6, 0xB, 0xC, 0xA);
		BLAKE256_GS_ALT(2, 7, 0x8, 0xD, 0xC);
		BLAKE256_GS_ALT(3, 4, 0x9, 0xE, 0xE);
	}
	BLAKE256_COMPRESS_END_DIRECT_NOSWAP(input_words[0], input_words[1], input_words[2], input_words[3], input_words[4], input_words[5], input_words[6], input_words[7], 
		                         out_words[0], out_words[1], out_words[2], out_words[3], out_words[4], out_words[5], out_words[6], out_words[7]);

	out_words[0] = sph_bswap32(out_words[0]);
	out_words[1] = sph_bswap32(out_words[1]);
	out_words[2] = sph_bswap32(out_words[2]);
	out_words[3] = sph_bswap32(out_words[3]);
	out_words[4] = sph_bswap32(out_words[4]);
	out_words[5] = sph_bswap32(out_words[5]);
	out_words[6] = sph_bswap32(out_words[6]);
	out_words[7] = sph_bswap32(out_words[7]);
}


// blake80 + keccak, in(80 bytes), out(32 bytes)
inline void blakeKeccak80_noswap(__constant const uint* in_words, uint gid, __global uint2* out_dwords, const uint isolate)
{
	//printf("INPUT WORDS[1]: %s\n", debug_print_hash((uint*)(input_words)));
	//printf("INPUT WORDS[2]: %s\n", debug_print_hash((uint*)((uint8_t*)(input_words) + (52 - 32))));
	BLAKE256_COMPRESS32_STATE;


#ifndef PRECALC_BLAKE
	//T0 = SPH_T32(T0 + 512);
	// Input == input words, already byteswapped
	uint input_words[19];
	for (uint i=0; i<20; i++) { input_words[i] = in_words[i]; }
	BLAKE256_COMPRESS_BEGIN_DIRECT(SPH_C32(0x6a09e667), SPH_C32(0xbb67ae85), SPH_C32(0x3c6ef372), SPH_C32(0xa54ff53a),  SPH_C32(0x510e527f), SPH_C32(0x9b05688c), SPH_C32(0x1f83d9ab), SPH_C32(0x5be0cd19),
		512, 0, (input_words[0]),(input_words[1]),(input_words[2]),(input_words[3]),(input_words[4]),(input_words[5]),(input_words[6]),(input_words[7]),(input_words[8]),(input_words[9]),(input_words[10]),(input_words[11]),(input_words[12]),(input_words[13]),(input_words[14]),(input_words[15]));
#pragma unroll 
	for (uint R = 0; R< BLAKE32_ROUNDS; R++) {
		BLAKE256_GS_ALT(0, 4, 0x8, 0xC, 0x0);
		BLAKE256_GS_ALT(1, 5, 0x9, 0xD, 0x2);
		BLAKE256_GS_ALT(2, 6, 0xA, 0xE, 0x4);
		BLAKE256_GS_ALT(3, 7, 0xB, 0xF, 0x6);
		BLAKE256_GS_ALT(0, 5, 0xA, 0xF, 0x8);
		BLAKE256_GS_ALT(1, 6, 0xB, 0xC, 0xA);
		BLAKE256_GS_ALT(2, 7, 0x8, 0xD, 0xC);
		BLAKE256_GS_ALT(3, 4, 0x9, 0xE, 0xE);
	}
	BLAKE256_COMPRESS_END_DIRECT_NOSWAP(SPH_C32(0x6a09e667), SPH_C32(0xbb67ae85), SPH_C32(0x3c6ef372), SPH_C32(0xa54ff53a),  SPH_C32(0x510e527f), SPH_C32(0x9b05688c), SPH_C32(0x1f83d9ab), SPH_C32(0x5be0cd19),
		                                input_words[0], input_words[1], input_words[2], input_words[3], input_words[4], input_words[5], input_words[6], input_words[7]);
#else
	__constant const uint* input_words = in_words;
#endif

	//printf("nonce[%u] blake32 after step H=[%x,%x,%x,%x,%x,%x,%x,%x]\n", gid, input_words[0], input_words[1], input_words[2], input_words[3], input_words[4], input_words[5], input_words[6], input_words[7]);

	// blake close - filled case
	//T0 -= 512 - 128; // i.e. 128
	//T0 = SPH_T32(T0 + 512);
	//T0 = 640;

	//printf("blake32 full step T0=0x%x T1=0x%x H=[%x,%x,%x,%x,%x,%x,%x,%x] S=[%x,%x,%x,%x]\n", T0, T1, H0, H1, H2, H3, H4, H5, H6, H7, S0, S1, S2, S3);

	// NOTE: At this stage, input_words[0...7] contains the previous state of the blake hash so we can continue on similarly to before

	BLAKE256_COMPRESS_BEGIN_DIRECT(input_words[0], input_words[1], input_words[2], input_words[3], input_words[4], input_words[5], input_words[6], input_words[7], 640, 0, 
		                          (input_words[16]),(input_words[17]),(input_words[18]),(gid),2147483648,0,0,0,0,0,0,0,0,1,0,640);
#pragma unroll 
	for (uint R = 0; R< BLAKE32_ROUNDS; R++) {
		BLAKE256_GS_ALT(0, 4, 0x8, 0xC, 0x0);
		BLAKE256_GS_ALT(1, 5, 0x9, 0xD, 0x2);
		BLAKE256_GS_ALT(2, 6, 0xA, 0xE, 0x4);
		BLAKE256_GS_ALT(3, 7, 0xB, 0xF, 0x6);
		BLAKE256_GS_ALT(0, 5, 0xA, 0xF, 0x8);
		BLAKE256_GS_ALT(1, 6, 0xB, 0xC, 0xA);
		BLAKE256_GS_ALT(2, 7, 0x8, 0xD, 0xC);
		BLAKE256_GS_ALT(3, 4, 0x9, 0xE, 0xE);
	}
	//BLAKE256_COMPRESS_END_DIRECT_NOSWAP(input_words[0], input_words[1], input_words[2], input_words[3], input_words[4], input_words[5], input_words[6], input_words[7], 
	//	                         out_words[0], out_words[1], out_words[2], out_words[3], out_words[4], out_words[5], out_words[6], out_words[7]);


	uint2 keccak_gpu_state[25] = {0}; // 50 ints

	keccak_gpu_state[0].x = sph_bswap32(input_words[0] ^ (V[0] ^ V[8]));
	keccak_gpu_state[0].y = sph_bswap32(input_words[1] ^ (V[1] ^ V[9]));
	keccak_gpu_state[1].x = sph_bswap32(input_words[2] ^ (V[2] ^ V[10]));
	keccak_gpu_state[1].y = sph_bswap32(input_words[3] ^ (V[3] ^ V[11]));
	keccak_gpu_state[2].x = sph_bswap32(input_words[4] ^ (V[4] ^ V[12]));
	keccak_gpu_state[2].y = sph_bswap32(input_words[5] ^ (V[5] ^ V[13]));
	keccak_gpu_state[3].x = sph_bswap32(input_words[6] ^ (V[6] ^ V[14]));
	keccak_gpu_state[3].y = sph_bswap32(input_words[7] ^ (V[7] ^ V[15]));
	keccak_gpu_state[4] = (uint2)(1, 0);

	keccak_gpu_state[16] = (uint2)(0, 0x80000000);
	
	keccak_block_uint2(&keccak_gpu_state[0], isolate);
	
	#pragma unroll 4
	for (int i = 0; i<4; i++) { out_dwords[i] = keccak_gpu_state[i]; }
}

// blake80, in(52 bytes), out(32 bytes)
inline void blake52(const uint* input_words, __global uint* out_words)
{
	// Blake256 vars
	//BLAKE256_STATE;
	BLAKE256_COMPRESS32_STATE;

	// Blake256 start hash
	//INIT_BLAKE256_STATE;
	// Blake hash full input
	// blake close - t0==0 case
	//T0 = SPH_C32(0xFFFFFE00) + 416;
	//T1 = SPH_C32(0xFFFFFFFF);
	//T0 = SPH_T32(T0 + 512);
	//T1 = SPH_T32(T1 + 1);

	//printf("blake32 full step T0=0x%x T1=0x%x H=[%x,%x,%x,%x,%x,%x,%x,%x] S=[%x,%x,%x,%x]\n", T0, T1, H0, H1, H2, H3, H4, H5, H6, H7, S0, S1, S2, S3);
	BLAKE256_COMPRESS_BEGIN_DIRECT(SPH_C32(0x6a09e667), SPH_C32(0xbb67ae85), SPH_C32(0x3c6ef372), SPH_C32(0xa54ff53a),  SPH_C32(0x510e527f), SPH_C32(0x9b05688c), SPH_C32(0x1f83d9ab), SPH_C32(0x5be0cd19),
		SPH_C32(0x1a0), SPH_C32(0x0), sph_bswap32(input_words[0]),sph_bswap32(input_words[1]),sph_bswap32(input_words[2]),sph_bswap32(input_words[3]),sph_bswap32(input_words[4]),sph_bswap32(input_words[5]),sph_bswap32(input_words[6]),sph_bswap32(input_words[7]),sph_bswap32(input_words[8]),sph_bswap32(input_words[9]),sph_bswap32(input_words[10]),sph_bswap32(input_words[11]),sph_bswap32(input_words[12]),2147483649,0,416);

	//BLAKE256_COMPRESS_BEGIN_LIGHT(SPH_C32(0x1a0), SPH_C32(0x1), 
	//	                          sph_bswap32(input_words[0]),sph_bswap32(input_words[1]),sph_bswap32(input_words[2]),sph_bswap32(input_words[3]),sph_bswap32(input_words[4]),sph_bswap32(input_words[5]),sph_bswap32(input_words[6]),sph_bswap32(input_words[7]),sph_bswap32(input_words[8]),sph_bswap32(input_words[9]),sph_bswap32(input_words[10]),sph_bswap32(input_words[11]),sph_bswap32(input_words[12]),2147483649,0,416);
#pragma unroll 
	for (uint R = 0; R< BLAKE32_ROUNDS; R++) {
		BLAKE256_GS_ALT(0, 4, 0x8, 0xC, 0x0);
		BLAKE256_GS_ALT(1, 5, 0x9, 0xD, 0x2);
		BLAKE256_GS_ALT(2, 6, 0xA, 0xE, 0x4);
		BLAKE256_GS_ALT(3, 7, 0xB, 0xF, 0x6);
		BLAKE256_GS_ALT(0, 5, 0xA, 0xF, 0x8);
		BLAKE256_GS_ALT(1, 6, 0xB, 0xC, 0xA);
		BLAKE256_GS_ALT(2, 7, 0x8, 0xD, 0xC);
		BLAKE256_GS_ALT(3, 4, 0x9, 0xE, 0xE);
	}
	//BLAKE256_COMPRESS_END;
	BLAKE256_COMPRESS_END_DIRECT_NOSWAP(SPH_C32(0x6a09e667), SPH_C32(0xbb67ae85), SPH_C32(0x3c6ef372), SPH_C32(0xa54ff53a),  SPH_C32(0x510e527f), SPH_C32(0x9b05688c), SPH_C32(0x1f83d9ab), SPH_C32(0x5be0cd19), out_words[0], out_words[1], out_words[2], out_words[3], out_words[4], out_words[5], out_words[6], out_words[7]);

	//printf("blake32 final step T0=0x%x T1=0x%x H=[%x,%x,%x,%x,%x,%x,%x,%x] S=[%x,%x,%x,%x]\n", T0, T1, H0, H1, H2, H3, H4, H5, H6, H7, S0, S1, S2, S3);

	out_words[0] = sph_bswap32(out_words[0]);
	out_words[1] = sph_bswap32(out_words[1]);
	out_words[2] = sph_bswap32(out_words[2]);
	out_words[3] = sph_bswap32(out_words[3]);
	out_words[4] = sph_bswap32(out_words[4]);
	out_words[5] = sph_bswap32(out_words[5]);
	out_words[6] = sph_bswap32(out_words[6]);
	out_words[7] = sph_bswap32(out_words[7]);
}


// blake80 + keccak, in(52 bytes), out(32 bytes)
inline void blakeKeccak52(const uint* input_words, __global uint2* out_dwords, const uint isolate)
{
	// Blake256 vars
	//BLAKE256_STATE;
	BLAKE256_COMPRESS32_STATE;

	// Blake256 start hash
	//INIT_BLAKE256_STATE;
	// Blake hash full input
	// blake close - t0==0 case
	//T0 = SPH_C32(0xFFFFFE00) + 416;
	//T1 = SPH_C32(0xFFFFFFFF);
	//T0 = SPH_T32(T0 + 512);
	//T1 = SPH_T32(T1 + 1);

	//printf("blake32 full step T0=0x%x T1=0x%x H=[%x,%x,%x,%x,%x,%x,%x,%x] S=[%x,%x,%x,%x]\n", T0, T1, H0, H1, H2, H3, H4, H5, H6, H7, S0, S1, S2, S3);
	BLAKE256_COMPRESS_BEGIN_DIRECT(SPH_C32(0x6a09e667), SPH_C32(0xbb67ae85), SPH_C32(0x3c6ef372), SPH_C32(0xa54ff53a),  SPH_C32(0x510e527f), SPH_C32(0x9b05688c), SPH_C32(0x1f83d9ab), SPH_C32(0x5be0cd19),
		SPH_C32(0x1a0), SPH_C32(0x0), sph_bswap32(input_words[0]),sph_bswap32(input_words[1]),sph_bswap32(input_words[2]),sph_bswap32(input_words[3]),sph_bswap32(input_words[4]),sph_bswap32(input_words[5]),sph_bswap32(input_words[6]),sph_bswap32(input_words[7]),sph_bswap32(input_words[8]),sph_bswap32(input_words[9]),sph_bswap32(input_words[10]),sph_bswap32(input_words[11]),sph_bswap32(input_words[12]),2147483649,0,416);

	//BLAKE256_COMPRESS_BEGIN_LIGHT(SPH_C32(0x1a0), SPH_C32(0x1), 
	//	                          sph_bswap32(input_words[0]),sph_bswap32(input_words[1]),sph_bswap32(input_words[2]),sph_bswap32(input_words[3]),sph_bswap32(input_words[4]),sph_bswap32(input_words[5]),sph_bswap32(input_words[6]),sph_bswap32(input_words[7]),sph_bswap32(input_words[8]),sph_bswap32(input_words[9]),sph_bswap32(input_words[10]),sph_bswap32(input_words[11]),sph_bswap32(input_words[12]),2147483649,0,416);
#pragma unroll 
	for (uint R = 0; R< BLAKE32_ROUNDS; R++) {
		BLAKE256_GS_ALT(0, 4, 0x8, 0xC, 0x0);
		BLAKE256_GS_ALT(1, 5, 0x9, 0xD, 0x2);
		BLAKE256_GS_ALT(2, 6, 0xA, 0xE, 0x4);
		BLAKE256_GS_ALT(3, 7, 0xB, 0xF, 0x6);
		BLAKE256_GS_ALT(0, 5, 0xA, 0xF, 0x8);
		BLAKE256_GS_ALT(1, 6, 0xB, 0xC, 0xA);
		BLAKE256_GS_ALT(2, 7, 0x8, 0xD, 0xC);
		BLAKE256_GS_ALT(3, 4, 0x9, 0xE, 0xE);
	}
	//BLAKE256_COMPRESS_END;
	//BLAKE256_COMPRESS_END_DIRECT_NOSWAP(SPH_C32(0x6a09e667), SPH_C32(0xbb67ae85), SPH_C32(0x3c6ef372), SPH_C32(0xa54ff53a),  SPH_C32(0x510e527f), SPH_C32(0x9b05688c), SPH_C32(0x1f83d9ab), SPH_C32(0x5be0cd19), out_words[0], out_words[1], out_words[2], out_words[3], out_words[4], out_words[5], out_words[6], out_words[7]);

	//printf("blake32 final step T0=0x%x T1=0x%x H=[%x,%x,%x,%x,%x,%x,%x,%x] S=[%x,%x,%x,%x]\n", T0, T1, H0, H1, H2, H3, H4, H5, H6, H7, S0, S1, S2, S3);

	uint2 keccak_gpu_state[25] = {0}; // 50 ints

	keccak_gpu_state[0].x = sph_bswap32(SPH_C32(0x6a09e667) ^ (V[0] ^ V[8]));
	keccak_gpu_state[0].y = sph_bswap32(SPH_C32(0xbb67ae85) ^ (V[1] ^ V[9]));
	keccak_gpu_state[1].x = sph_bswap32(SPH_C32(0x3c6ef372) ^ (V[2] ^ V[10]));
	keccak_gpu_state[1].y = sph_bswap32(SPH_C32(0xa54ff53a) ^ (V[3] ^ V[11]));
	keccak_gpu_state[2].x = sph_bswap32(SPH_C32(0x510e527f) ^ (V[4] ^ V[12]));
	keccak_gpu_state[2].y = sph_bswap32(SPH_C32(0x9b05688c) ^ (V[5] ^ V[13]));
	keccak_gpu_state[3].x = sph_bswap32(SPH_C32(0x1f83d9ab) ^ (V[6] ^ V[14]));
	keccak_gpu_state[3].y = sph_bswap32(SPH_C32(0x5be0cd19) ^ (V[7] ^ V[15]));
	keccak_gpu_state[4] = (uint2)(1, 0);

	keccak_gpu_state[16] = (uint2)(0, 0x80000000);
	
	keccak_block_uint2(&keccak_gpu_state[0], isolate);
	
	#pragma unroll 4
	for (int i = 0; i<4; i++) { out_dwords[i] = keccak_gpu_state[i]; }
}

// skein32, in(32 bytes), out(32 bytes)
inline void skein32(__global const ulong* in_dwords, __global ulong* out_dwords)
{
	// in_dwords could be rolled into p*
	//printf("skein32 in=%s\n", debug_print_hash(in_dwords));

	ulong h[9];
	ulong t[3];
	ulong dt0,dt1,dt2,dt3;
	ulong p0, p1, p2, p3, p4, p5, p6, p7;
	h[8] = skein_ks_parity;

	for (int i = 0; i<8; i++) {
		h[i] = SKEIN_IV512_256[i];
		h[8] ^= h[i];
	}

	t[0]=skein_t12[0];
	t[1]=skein_t12[1];
	t[2]=skein_t12[2];

	dt0= (in_dwords[0]);
	dt1= (in_dwords[1]);
	dt2= (in_dwords[2]);
	dt3= (in_dwords[3]);

	//printf("Skein in hash=%lu,%lu,%lu,%lu\n",
	//	dt0, dt1, dt2, dt3);

	p0 = h[0] + dt0;
	p1 = h[1] + dt1;
	p2 = h[2] + dt2;
	p3 = h[3] + dt3;
	p4 = h[4];
	p5 = h[5] + t[0];
	p6 = h[6] + t[1];
	p7 = h[7];

	#pragma unroll 
	for (int i = 1; i<19; i+=2) {Round_8_512(p0,p1,p2,p3,p4,p5,p6,p7,i);}
	
	p0 ^= dt0;
	p1 ^= dt1;
	p2 ^= dt2;
	p3 ^= dt3;

	h[0] = p0;
	h[1] = p1;
	h[2] = p2;
	h[3] = p3;
	h[4] = p4;
	h[5] = p5;
	h[6] = p6;
	h[7] = p7;
	h[8] = skein_ks_parity;

	for (int i = 0; i<8; i++) { h[8] ^= h[i]; }
		
	t[0] = skein_t12[3];
	t[1] = skein_t12[4];
	t[2] = skein_t12[5];
	p5 += t[0];  //p5 already equal h[5] 
	p6 += t[1];

    #pragma unroll
	for (int i = 1; i<19; i+=2) { Round_8_512(p0, p1, p2, p3, p4, p5, p6, p7, i); }


	//printf("skein out regs =%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu\n", p0, p1, p2, p3, p4, p5, p6, p7);

	out_dwords[0]      = (p0);
	out_dwords[1]      = (p1);
	out_dwords[2]      = (p2);
	out_dwords[3]      = (p3);
}

// bmw32, in(32 bytes), out(32 bytes)
inline void bmw32_to_global(__global const uint* in_words, __global uint* out_words)
{
	//printf("bmw32 in bytes=%s\n", debug_print_hash(in_hash));

	uint dh[16] = {
		0x40414243, 0x44454647,
		0x48494A4B, 0x4C4D4E4F,
		0x50515253, 0x54555657,
		0x58595A5B, 0x5C5D5E5F,
		0x60616263, 0x64656667,
		0x68696A6B, 0x6C6D6E6F,
		0x70717273, 0x74757677,
		0x78797A7B, 0x7C7D7E7F
	};

	uint message[16];
	for (int i = 0; i<8; i++) message[i] = (in_words[i]);
	for (int i = 9; i<14; i++) message[i] = 0;
	message[8]= 0x80;
	message[14]=0x100;
	message[15]=0;

	Compression256(message, dh);
	message[0] = 0xaaaaaaa0;
	message[1] = 0xaaaaaaa1;
	message[2] = 0xaaaaaaa2;
	message[3] = 0xaaaaaaa3;
	message[4] = 0xaaaaaaa4;
	message[5] = 0xaaaaaaa5;
	message[6] = 0xaaaaaaa6;
	message[7] = 0xaaaaaaa7;
	message[8] = 0xaaaaaaa8;
	message[9] = 0xaaaaaaa9;
	message[10] = 0xaaaaaaaa;
	message[11] = 0xaaaaaaab;
	message[12] = 0xaaaaaaac;
	message[13] = 0xaaaaaaad;
	message[14] = 0xaaaaaaae;
	message[15] = 0xaaaaaaaf;
	Compression256(dh, message);

	#pragma unroll
	for (int i=8; i<16; i++) {
		out_words[i-8] = (message[i]);
	}
}


#ifdef BMW32_ONLY_RETURN_LAST
inline ulong bmw32(__global const uint* in_words)
#else
inline void bmw32(__global const uint* in_words, uint* out_words)
#endif
{
	//printf("bmw32 in bytes=%s\n", debug_print_hash(in_hash));

	uint dh[16] = {
		0x40414243, 0x44454647,
		0x48494A4B, 0x4C4D4E4F,
		0x50515253, 0x54555657,
		0x58595A5B, 0x5C5D5E5F,
		0x60616263, 0x64656667,
		0x68696A6B, 0x6C6D6E6F,
		0x70717273, 0x74757677,
		0x78797A7B, 0x7C7D7E7F
	};

	uint message[16];
	for (int i = 0; i<8; i++) message[i] = (in_words[i]);
	for (int i = 9; i<14; i++) message[i] = 0;
	message[8]= 0x80;
	message[14]=0x100;
	message[15]=0;

	Compression256(message, dh);
	message[0] = 0xaaaaaaa0;
	message[1] = 0xaaaaaaa1;
	message[2] = 0xaaaaaaa2;
	message[3] = 0xaaaaaaa3;
	message[4] = 0xaaaaaaa4;
	message[5] = 0xaaaaaaa5;
	message[6] = 0xaaaaaaa6;
	message[7] = 0xaaaaaaa7;
	message[8] = 0xaaaaaaa8;
	message[9] = 0xaaaaaaa9;
	message[10] = 0xaaaaaaaa;
	message[11] = 0xaaaaaaab;
	message[12] = 0xaaaaaaac;
	message[13] = 0xaaaaaaad;
	message[14] = 0xaaaaaaae;
	message[15] = 0xaaaaaaaf;
#ifdef BMW32_ONLY_RETURN_LAST
	return Compression256_last(dh, message);
#else
	Compression256(dh, message);

	#pragma unroll
	for (int i=8; i<16; i++) {
		out_words[i-8] = (message[i]);
	}
#endif
}

// cubehash32, in(32 bytes), out(32 bytes)
inline void cubehash32(__global const uint* in_words, __global uint* out_words)
{
	uint x[32]; // NOTE: in this case, a contiguous array is basically identical to using lots of individual vars 

	// IDEA: absorb out_words into this
	x[0x0] =  0xEA2BD4B4 ^ in_words[0]; x[0x1] =  0xCCD6F29F ^ in_words[1]; x[0x2] =  0x63117E71 ^ in_words[2];
	x[0x3] =  0x35481EAE ^ in_words[3]; x[0x4] =  0x22512D5B ^ in_words[4]; x[0x5] =  0xE5D94E63 ^ in_words[5];
	x[0x6] =  0x7E624131 ^ in_words[6]; x[0x7] =  0xF4CC12BE ^ in_words[7]; x[0x8] =  0xC2D0B696;
	x[0x9] =  0x42AF2070; x[0xa] =  0xD0720C35; x[0xb] =  0x3361DA8C;
	x[0xc] =  0x28CCECA4; x[0xd] =  0x8EF8AD83; x[0xe] =  0x4680AC00;
	x[0xf] =  0x40E5FBAB;

	x[0x10] =  0xD89041C3; x[0x11] =  0x6107FBD5;
	x[0x12] =  0x6C859D41; x[0x13] =  0xF0B26679; x[0x14] =  0x09392549;
	x[0x15] =  0x5FA25603; x[0x16] =  0x65C892FD; x[0x17] =  0x93CB6285;
	x[0x18] =  0x2AF2B5AE; x[0x19] =  0x9E4B4E60; x[0x1a] =  0x774ABFDD;
	x[0x1b] =  0x85254725; x[0x1c] =  0x15815AEB; x[0x1d] =  0x4AB6AAD6;
	x[0x1e] =  0x9CDAF8AF; x[0x1f] =  0xD6032C0A;

	SIXTEEN_ROUNDS;
	x[0x0] ^= 0x80;
	SIXTEEN_ROUNDS;
	x[0x1f] ^= 0x01;
	for (int i = 0; i < 10; ++i) SIXTEEN_ROUNDS;

	out_words[0] = x[0];
	out_words[1] = x[1];
	out_words[2] = x[2];
	out_words[3] = x[3];
	out_words[4] = x[4];
	out_words[5] = x[5];
	out_words[6] = x[6];
	out_words[7] = x[7];
}

// run-of-the-mill lyra2
inline void lyra2(__global const ulong* in_dwords, __global ulong* out_dwords,LYRA_SCOPE ulong4* DMatrix)
{
	ulong4 state[4];

	state[0].x = in_dwords[0]; //password
	state[0].y = in_dwords[1]; //password
	state[0].z = in_dwords[2]; //password
	state[0].w = in_dwords[3]; //password
	state[1] = state[0];
	state[2] = SPH_ULONG4(0x6a09e667f3bcc908UL, 0xbb67ae8584caa73bUL, 0x3c6ef372fe94f82bUL, 0xa54ff53a5f1d36f1UL);
	state[3] = SPH_ULONG4(0x510e527fade682d1UL, 0x9b05688c2b3e6c1fUL, 0x1f83d9abfb41bd6bUL, 0x5be0cd19137e2179UL);
	for (int i = 0; i<12; i++) { round_lyra(state); } 

	state[0] ^= SPH_ULONG4(0x20,0x20,0x20,0x01);
	state[1] ^= SPH_ULONG4(0x04,0x04,0x80,0x0100000000000000);

	for (int i = 0; i<12; i++) { round_lyra(state); } 


	const uint ps1 = (memshift * 3);
	//#pragma unroll 4
	for (int i = 0; i < 4; i++)
	{
		uint s1 = ps1 - memshift * i;
		for (int j = 0; j < 3; j++)
			(DMatrix)[j+s1] = state[j];

		round_lyra(state);
	}

	ulong4 state1[3];
	reduceDuplexf(state1, state,DMatrix);

	ulong4 state2[3];
	reduceDuplexRowSetupf(state1, state2, 1, 0, 2, state,DMatrix);
	reduceDuplexRowSetupf(state1, state2, 2, 1, 3, state,DMatrix);


	uint rowa;
	uint prev = 3;
	for (uint i = 0; i<4; i++) {
		rowa = state[0].x & 3;
		reduceDuplexRowf(state1, state2,prev, rowa, i, state, DMatrix);
		prev = i;
	}

	const uint shift = (memshift * 4 * rowa);

	for (int j = 0; j < 3; j++)
		state[j] ^= (DMatrix)[j+shift];

	for (int i = 0; i < 12; i++)
		round_lyra(state);
	
	//////////////////////////////////////

	for (int i = 0; i<4; i++) {out_dwords[i] = ((ulong*)state)[i];} 
}

// Perform header mix, outputting the mix hash
inline uint4 hashimoto_mix(uint* headerHash, __global const uint16 *dag, const ulong n)
{
	const ulong mixhashes = MIX_BYTES / HASH_BYTES;    // 2
	MixNodes mix;                  // 64 bytes
	mix.nodes8[0] = mix.nodes8[1] = ((uint8*)headerHash)[0];
	uint header_int = mix.values[0];
	
	for (uint i = 0; i < ACCESSES; i++) {
		// Pick a dag index to mix with
		const uint p = fnv(i ^ header_int, mix.values[i % 16]) % (n / mixhashes);

		// Mix in mixer. Note dag access is at most aligned to 64 bytes
		mix.nodes16 *= FNV_PRIME;
		mix.nodes16 ^= dag[p];
	}
	
	uint4 mixhash;
	mixhash.x = fnv(fnv(fnv(mix.values[0],  mix.values[0 + 1]),  mix.values[0 + 2]),  mix.values[0 + 3]);
	mixhash.y = fnv(fnv(fnv(mix.values[4],  mix.values[4 + 1]),  mix.values[4 + 2]),  mix.values[4 + 3]);
	mixhash.z = fnv(fnv(fnv(mix.values[8],  mix.values[8 + 1]),  mix.values[8 + 2]),  mix.values[8 + 3]);
	mixhash.w = fnv(fnv(fnv(mix.values[12], mix.values[12 + 1]), mix.values[12 + 2]), mix.values[12 + 3]);
	return mixhash;
}

#define HASH_WORDS 8
#define MIX_WORDS 16

// Set to enable hash testing kernel variant
//#define TEST_KERNEL_HASH


__attribute__((reqd_work_group_size(WORKSIZE, 1, 1)))
__kernel void search(
	 __global hash32_t* hashes,
	__constant uint const* g_header,
  uint isolate)
{
 	const uint gid = get_global_id(0);
 	__global hash32_t *hash = (__global hash32_t *)(hashes + ((gid % MAX_GLOBAL_THREADS)));

    blakeKeccak80_noswap(g_header, gid, hash->u2, isolate);
}

__attribute__((reqd_work_group_size(WORKSIZE, 1, 1)))
__kernel void search1(
	 __global hash32_t* hashes )
{
 	const uint gid = get_global_id(0);
 	__global hash32_t *hash = (__global hash32_t *)(hashes + ((gid % MAX_GLOBAL_THREADS)));

    cubehash32(hash->h4, hash->h4);
}

__attribute__((reqd_work_group_size(LYRA_WORKSIZE, 1, 1)))
__kernel void search2(
	 __global hash32_t* hashes,
	__global ulong4* g_lyre_nodes )
{
 	const uint gid = get_global_id(0);
	const uint hash_output_idx = gid;// - get_global_offset(0);
	 __global hash32_t *hash = (__global hash32_t *)(hashes + ((gid % MAX_GLOBAL_THREADS)));
	
	__global ulong4 *DMatrix = (__global ulong4 *)(g_lyre_nodes + (LYRA_SCRATCHBUF_SIZE_ULONG4 * (hash_output_idx % MAX_GLOBAL_THREADS)));

    lyra2(hash->h8, hash->h8, DMatrix);
}

__attribute__((reqd_work_group_size(WORKSIZE, 1, 1)))
__kernel void search3(
	 __global hash32_t* hashes )
{
 	const uint gid = get_global_id(0);
 	__global hash32_t *hash = (__global hash32_t *)(hashes + ((gid % MAX_GLOBAL_THREADS)));

    skein32(hash->h8, hash->h8);
}

__attribute__((reqd_work_group_size(WORKSIZE, 1, 1)))
__kernel void search4(
	 __global hash32_t* hashes )
{
 	const uint gid = get_global_id(0);
 	__global hash32_t *hash = (__global hash32_t *)(hashes + ((gid % MAX_GLOBAL_THREADS)));

    cubehash32(hash->h4, hash->h4);
}

__attribute__((reqd_work_group_size(WORKSIZE, 1, 1)))
__kernel void search5(
	 __global hash32_t* hashes )
{
 	const uint gid = get_global_id(0);
 	__global hash32_t *hash = (__global hash32_t *)(hashes + ((gid % MAX_GLOBAL_THREADS)));

    bmw32_to_global(hash->h4, hash->h4);
}


__attribute__((reqd_work_group_size(WORKSIZE, 1, 1)))
__kernel void search6(
	 __global hash32_t* hashes,
	__global uint16* const dag,
	const ulong DAG_ITEM_COUNT,
	const uint height,
  uint isolate)
{
 	const uint gid = get_global_id(0);
	 __global hash32_t *hash = (__global hash32_t *)(hashes + ((gid % MAX_GLOBAL_THREADS)));

	uint blockToHash[13];
#pragma unroll
	for (uint i=0; i<8; i++) {
		blockToHash[i] = hash->h4[i];
	}
  // Mix
  {
		uint4 mixHash = hashimoto_mix(blockToHash, dag, DAG_ITEM_COUNT);
		blockToHash[8] = height;
		blockToHash[9] = mixHash.x;
		blockToHash[10] = mixHash.y;
		blockToHash[11] = mixHash.z;
		blockToHash[12] = mixHash.w;
	}

	blakeKeccak52(blockToHash, hash->u2, isolate);
	//blake52(blockToHash, hash->h4);
}

__attribute__((reqd_work_group_size(WORKSIZE, 1, 1)))
__kernel void search7(
	 __global hash32_t* hashes )
{
 	const uint gid = get_global_id(0);
 	__global hash32_t *hash = (__global hash32_t *)(hashes + ((gid % MAX_GLOBAL_THREADS)));

    cubehash32(hash->h4, hash->h4);
}


__attribute__((reqd_work_group_size(LYRA_WORKSIZE, 1, 1)))
__kernel void search8(
	 __global hash32_t* hashes,
	__global ulong4* g_lyre_nodes )
{
 	const uint gid = get_global_id(0);
	const uint hash_output_idx = gid;// - get_global_offset(0);
	 __global hash32_t *hash = (__global hash32_t *)(hashes + ((gid % MAX_GLOBAL_THREADS)));
	
	__global ulong4 *DMatrix = (__global ulong4 *)(g_lyre_nodes + (LYRA_SCRATCHBUF_SIZE_ULONG4 * (hash_output_idx % MAX_GLOBAL_THREADS)));
	
    lyra2(hash->h8, hash->h8, DMatrix);
}

__attribute__((reqd_work_group_size(WORKSIZE, 1, 1)))
__kernel void search9(
	 __global hash32_t* hashes )
{
 	const uint gid = get_global_id(0);
 	__global hash32_t *hash = (__global hash32_t *)(hashes + ((gid % MAX_GLOBAL_THREADS)));

    skein32(hash->h8, hash->h8);
}


__attribute__((reqd_work_group_size(WORKSIZE, 1, 1)))
__kernel void search10(
	 __global hash32_t* hashes )
{
 	const uint gid = get_global_id(0);
 	__global hash32_t *hash = (__global hash32_t *)(hashes + ((gid % MAX_GLOBAL_THREADS)));

    cubehash32(hash->h4, hash->h4);
}


#ifndef TEST_KERNEL_HASH

__attribute__((reqd_work_group_size(WORKSIZE, 1, 1)))
__kernel void search11(
	 __global hash32_t* hashes,
	__global volatile uint* restrict g_output,
	const ulong target )
{
 	const uint gid = get_global_id(0);
	const uint hash_output_idx = gid;// - get_global_offset(0);
	 __global hash32_t *hash = (__global hash32_t *)(hashes + ((gid % MAX_GLOBAL_THREADS)));


#ifdef BMW32_ONLY_RETURN_LAST
	const ulong targetHashCheck = bmw32(hash->h4);
#else
	uint blockToHash[8];
	bmw32(hash->h4, blockToHash);
	const ulong targetHashCheck = ((ulong*)blockToHash)[3];
#endif

    //if (gid < 8) printf("Nonce[%u] BLOCK {0x%08x,0x%08x,0x%08x,0x%08x,0x%08x,0x%08x,0x%08x,0x%08x}\n", gid,
	//	   blockToHash[0], blockToHash[1], blockToHash[2], blockToHash[3], blockToHash[4], blockToHash[5], blockToHash[6], blockToHash[7]);

    // target itself should be in little-endian format, 
#ifdef NVIDIA

	if (targetHashCheck <= target)
	{
		//printf("Nonce %u Found target, %lx <= %lx\n", gid, out_long[3], target);
		uint slot = atomic_inc(&g_output[MAX_OUTPUTS]);
		//uint2 tgt = as_uint2(target);
		//printf("candidate %u => %08x %08x < %08x\n", slot, state[0].x, state[0].y, (uint) (target>>32));
		g_output[slot & MAX_OUTPUTS] = gid;
	}
#else

	if (targetHashCheck <= target)
	{
		//printf("Nonce %u Found target, %lx <= %lx BLOCK {0x%08x,0x%08x,0x%08x,0x%08x,0x%08x,0x%08x,0x%08x,0x%08x}\n", gid, block[7], target,
		//	block[0], block[1], block[2], block[3], block[4], block[5], block[6], block[7]);
		uint slot = min(MAX_OUTPUTS-1u, convert_uint(atomic_inc(&g_output[MAX_OUTPUTS])));
		g_output[slot] = gid;
	}
#endif
}

#else

__attribute__((reqd_work_group_size(WORKSIZE, 1, 1)))
__kernel void search11(
	 __global hash32_t* hashes,
	__global volatile uint* restrict g_output,
	const ulong target )
{
 	const uint gid = get_global_id(0);
	const uint hash_output_idx = gid - get_global_offset(0);
	 __global hash32_t *hash = (__global hash32_t *)(hashes + ((gid) % MAX_GLOBAL_THREADS)));

	uint blockToHash[8];
    bmw32(hash->h4, blockToHash);

	g_output[hash_output_idx].h4[0] = blockToHash[0];
	g_output[hash_output_idx].h4[1] = blockToHash[1];
	g_output[hash_output_idx].h4[2] = blockToHash[2];
	g_output[hash_output_idx].h4[3] = blockToHash[3];
	g_output[hash_output_idx].h4[4] = blockToHash[4];
	g_output[hash_output_idx].h4[5] = blockToHash[5];
	g_output[hash_output_idx].h4[6] = blockToHash[6];
	g_output[hash_output_idx].h4[7] = blockToHash[7];
}

#endif

#ifndef COMPILE_MAIN_ONLY

__kernel void GenerateDAG(uint start, __global const uint16 *_Cache, __global uint16 *_DAG, uint LIGHT_SIZE)
{
	__global const Node *Cache = (__global const Node *) _Cache;
	__global Node *DAG = (__global Node *) _DAG;
	uint NodeIdx = start + get_global_id(0);

	Node DAGNode = Cache[NodeIdx % LIGHT_SIZE];
	DAGNode.dwords[0] ^= NodeIdx;

	BLAKE256_STATE;
	BLAKE256_COMPRESS32_STATE;

	//printf("generateDAG %u\n", NodeIdx);

	// Apply blake to DAGNode

	INIT_BLAKE256_STATE;
	// Blake hash full input
	// blake close - t0==0 case
	T0 = SPH_C32(0xFFFFFE00) + 256;
	T1 = SPH_C32(0xFFFFFFFF);
	T0 = SPH_T32(T0 + 512);
	T1 = SPH_T32(T1 + 1);

	BLAKE256_COMPRESS32(sph_bswap32(DAGNode.dwords[0]),sph_bswap32(DAGNode.dwords[1]),sph_bswap32(DAGNode.dwords[2]),sph_bswap32(DAGNode.dwords[3]),sph_bswap32(DAGNode.dwords[4]),sph_bswap32(DAGNode.dwords[5]),sph_bswap32(DAGNode.dwords[6]),sph_bswap32(DAGNode.dwords[7]),2147483648,0,0,0,0,1,0,256);
	DAGNode.dwords[0] = sph_bswap32(H0);
	DAGNode.dwords[1] = sph_bswap32(H1);
	DAGNode.dwords[2] = sph_bswap32(H2);
	DAGNode.dwords[3] = sph_bswap32(H3);
	DAGNode.dwords[4] = sph_bswap32(H4);
	DAGNode.dwords[5] = sph_bswap32(H5);
	DAGNode.dwords[6] = sph_bswap32(H6);
	DAGNode.dwords[7] = sph_bswap32(H7);

	for (uint parent = 0; parent < DATASET_PARENTS; ++parent)
	{
		// Calculate parent
		uint ParentIdx = fnv(NodeIdx ^ parent, DAGNode.dwords[parent & 7]) % LIGHT_SIZE; // NOTE: LIGHT_SIZE == items, &7 == %8
		__global const Node *ParentNode = Cache + ParentIdx;

		#pragma unroll
		for (uint x = 0; x < 2; ++x)
		{
			// NOTE: fnv, we're basically operating on 4 ints at a time here
			DAGNode.dqwords[x] *= (uint4)(FNV_PRIME);
			DAGNode.dqwords[x] ^= ParentNode->dwords[0];
			DAGNode.dqwords[x] %= SPH_C32(0xffffffff);
		}
	}
	
	// Apply final blake to NodeIdx

	INIT_BLAKE256_STATE;
	// Blake hash full input
	// blake close - t0==0 case
	T0 = SPH_C32(0xFFFFFE00) + 256;
	T1 = SPH_C32(0xFFFFFFFF);
	T0 = SPH_T32(T0 + 512);
	T1 = SPH_T32(T1 + 1);

	BLAKE256_COMPRESS32(sph_bswap32(DAGNode.dwords[0]),sph_bswap32(DAGNode.dwords[1]),sph_bswap32(DAGNode.dwords[2]),sph_bswap32(DAGNode.dwords[3]),sph_bswap32(DAGNode.dwords[4]),sph_bswap32(DAGNode.dwords[5]),sph_bswap32(DAGNode.dwords[6]),sph_bswap32(DAGNode.dwords[7]),2147483648,0,0,0,0,1,0,256);
	DAGNode.dwords[0] = sph_bswap32(H0);
	DAGNode.dwords[1] = sph_bswap32(H1);
	DAGNode.dwords[2] = sph_bswap32(H2);
	DAGNode.dwords[3] = sph_bswap32(H3);
	DAGNode.dwords[4] = sph_bswap32(H4);
	DAGNode.dwords[5] = sph_bswap32(H5);
	DAGNode.dwords[6] = sph_bswap32(H6);
	DAGNode.dwords[7] = sph_bswap32(H7);


	DAG[NodeIdx] = DAGNode;
}

#endif

