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

#ifndef MAX_GLOBAL_THREADS
#define MAX_GLOBAL_THREADS 64
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

typedef union _Node
{
	uint dwords[8];
	uint4 dqwords[2];
} Node;

typedef union _MixNodes {
	uint values[16];
	uint16 nodes16;
} MixNodes;

// Output hash
typedef union {
	unsigned char h1[32];
	uint h4[8];
	ulong h8[4];
} hash32_t;

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
#define ROTL64_1(x, y)  amd_bitalign((x), (x).s10, (32U - y))
#define ROTL64_2(x, y)  amd_bitalign((x).s10, (x), (32U - y))
#define ROTL64_8(x, y)  amd_bitalign((x), (x).s10, 24U)
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

#define sph_bswap32(n) (rotate(n & 0x00FF00FF, 24U)|(rotate(n, 8U) & 0x00FF00FF))

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
for (uint R=0; R< 14; R++) { \
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

//
// BEGIN SKEIN256
//

__constant static const ulong SKEIN_IV512[] = {
  SPH_C64(0x4903ADFF749C51CE), SPH_C64(0x0D95DE399746DF03),
  SPH_C64(0x8FD1934127C79BCE), SPH_C64(0x9A255629FF352CB1),
  SPH_C64(0x5DB62599DF6CA7B0), SPH_C64(0xEABE394CA9D5C3F4),
  SPH_C64(0x991112C71A75B523), SPH_C64(0xAE18A40B660FCC33)
};

__constant static const ulong SKEIN_IV512_256[8] = {
	0xCCD044A12FDB3E13UL, 0xE83590301A79A9EBUL,
	0x55AEA0614F816E6FUL, 0x2A2767A4AE9B94DBUL,
	0xEC06025E74DD7683UL, 0xE7A436CDC4746251UL,
	0xC36FBAF9393AD185UL, 0x3EEDBA1833EDFC13UL
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

__constant static const ulong t12[6] =
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

//
// BEGIN CUBEHASH
//

#if !defined SPH_CUBEHASH_UNROLL
#define SPH_CUBEHASH_UNROLL   8
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
    xg = T32(x0 + xg); \
    x0 = ROTL32(x0, 7); \
    xh = T32(x1 + xh); \
    x1 = ROTL32(x1, 7); \
    xi = T32(x2 + xi); \
    x2 = ROTL32(x2, 7); \
    xj = T32(x3 + xj); \
    x3 = ROTL32(x3, 7); \
    xk = T32(x4 + xk); \
    x4 = ROTL32(x4, 7); \
    xl = T32(x5 + xl); \
    x5 = ROTL32(x5, 7); \
    xm = T32(x6 + xm); \
    x6 = ROTL32(x6, 7); \
    xn = T32(x7 + xn); \
    x7 = ROTL32(x7, 7); \
    xo = T32(x8 + xo); \
    x8 = ROTL32(x8, 7); \
    xp = T32(x9 + xp); \
    x9 = ROTL32(x9, 7); \
    xq = T32(xa + xq); \
    xa = ROTL32(xa, 7); \
    xr = T32(xb + xr); \
    xb = ROTL32(xb, 7); \
    xs = T32(xc + xs); \
    xc = ROTL32(xc, 7); \
    xt = T32(xd + xt); \
    xd = ROTL32(xd, 7); \
    xu = T32(xe + xu); \
    xe = ROTL32(xe, 7); \
    xv = T32(xf + xv); \
    xf = ROTL32(xf, 7); \
    x8 ^= xg; \
    x9 ^= xh; \
    xa ^= xi; \
    xb ^= xj; \
    xc ^= xk; \
    xd ^= xl; \
    xe ^= xm; \
    xf ^= xn; \
    x0 ^= xo; \
    x1 ^= xp; \
    x2 ^= xq; \
    x3 ^= xr; \
    x4 ^= xs; \
    x5 ^= xt; \
    x6 ^= xu; \
    x7 ^= xv; \
    xi = T32(x8 + xi); \
    x8 = ROTL32(x8, 11); \
    xj = T32(x9 + xj); \
    x9 = ROTL32(x9, 11); \
    xg = T32(xa + xg); \
    xa = ROTL32(xa, 11); \
    xh = T32(xb + xh); \
    xb = ROTL32(xb, 11); \
    xm = T32(xc + xm); \
    xc = ROTL32(xc, 11); \
    xn = T32(xd + xn); \
    xd = ROTL32(xd, 11); \
    xk = T32(xe + xk); \
    xe = ROTL32(xe, 11); \
    xl = T32(xf + xl); \
    xf = ROTL32(xf, 11); \
    xq = T32(x0 + xq); \
    x0 = ROTL32(x0, 11); \
    xr = T32(x1 + xr); \
    x1 = ROTL32(x1, 11); \
    xo = T32(x2 + xo); \
    x2 = ROTL32(x2, 11); \
    xp = T32(x3 + xp); \
    x3 = ROTL32(x3, 11); \
    xu = T32(x4 + xu); \
    x4 = ROTL32(x4, 11); \
    xv = T32(x5 + xv); \
    x5 = ROTL32(x5, 11); \
    xs = T32(x6 + xs); \
    x6 = ROTL32(x6, 11); \
    xt = T32(x7 + xt); \
    x7 = ROTL32(x7, 11); \
    xc ^= xi; \
    xd ^= xj; \
    xe ^= xg; \
    xf ^= xh; \
    x8 ^= xm; \
    x9 ^= xn; \
    xa ^= xk; \
    xb ^= xl; \
    x4 ^= xq; \
    x5 ^= xr; \
    x6 ^= xo; \
    x7 ^= xp; \
    x0 ^= xu; \
    x1 ^= xv; \
    x2 ^= xs; \
    x3 ^= xt; \
  } while (0)

#define ROUND_ODD   do { \
    xj = T32(xc + xj); \
    xc = ROTL32(xc, 7); \
    xi = T32(xd + xi); \
    xd = ROTL32(xd, 7); \
    xh = T32(xe + xh); \
    xe = ROTL32(xe, 7); \
    xg = T32(xf + xg); \
    xf = ROTL32(xf, 7); \
    xn = T32(x8 + xn); \
    x8 = ROTL32(x8, 7); \
    xm = T32(x9 + xm); \
    x9 = ROTL32(x9, 7); \
    xl = T32(xa + xl); \
    xa = ROTL32(xa, 7); \
    xk = T32(xb + xk); \
    xb = ROTL32(xb, 7); \
    xr = T32(x4 + xr); \
    x4 = ROTL32(x4, 7); \
    xq = T32(x5 + xq); \
    x5 = ROTL32(x5, 7); \
    xp = T32(x6 + xp); \
    x6 = ROTL32(x6, 7); \
    xo = T32(x7 + xo); \
    x7 = ROTL32(x7, 7); \
    xv = T32(x0 + xv); \
    x0 = ROTL32(x0, 7); \
    xu = T32(x1 + xu); \
    x1 = ROTL32(x1, 7); \
    xt = T32(x2 + xt); \
    x2 = ROTL32(x2, 7); \
    xs = T32(x3 + xs); \
    x3 = ROTL32(x3, 7); \
    x4 ^= xj; \
    x5 ^= xi; \
    x6 ^= xh; \
    x7 ^= xg; \
    x0 ^= xn; \
    x1 ^= xm; \
    x2 ^= xl; \
    x3 ^= xk; \
    xc ^= xr; \
    xd ^= xq; \
    xe ^= xp; \
    xf ^= xo; \
    x8 ^= xv; \
    x9 ^= xu; \
    xa ^= xt; \
    xb ^= xs; \
    xh = T32(x4 + xh); \
    x4 = ROTL32(x4, 11); \
    xg = T32(x5 + xg); \
    x5 = ROTL32(x5, 11); \
    xj = T32(x6 + xj); \
    x6 = ROTL32(x6, 11); \
    xi = T32(x7 + xi); \
    x7 = ROTL32(x7, 11); \
    xl = T32(x0 + xl); \
    x0 = ROTL32(x0, 11); \
    xk = T32(x1 + xk); \
    x1 = ROTL32(x1, 11); \
    xn = T32(x2 + xn); \
    x2 = ROTL32(x2, 11); \
    xm = T32(x3 + xm); \
    x3 = ROTL32(x3, 11); \
    xp = T32(xc + xp); \
    xc = ROTL32(xc, 11); \
    xo = T32(xd + xo); \
    xd = ROTL32(xd, 11); \
    xr = T32(xe + xr); \
    xe = ROTL32(xe, 11); \
    xq = T32(xf + xq); \
    xf = ROTL32(xf, 11); \
    xt = T32(x8 + xt); \
    x8 = ROTL32(x8, 11); \
    xs = T32(x9 + xs); \
    x9 = ROTL32(x9, 11); \
    xv = T32(xa + xv); \
    xa = ROTL32(xa, 11); \
    xu = T32(xb + xu); \
    xb = ROTL32(xb, 11); \
    x0 ^= xh; \
    x1 ^= xg; \
    x2 ^= xj; \
    x3 ^= xi; \
    x4 ^= xl; \
    x5 ^= xk; \
    x6 ^= xn; \
    x7 ^= xm; \
    x8 ^= xp; \
    x9 ^= xo; \
    xa ^= xr; \
    xb ^= xq; \
    xc ^= xt; \
    xd ^= xs; \
    xe ^= xv; \
    xf ^= xu; \
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
    int j; \
    for (j = 0; j < 8; j ++) { \
      ROUND_EVEN; \
      ROUND_ODD; \
    } \
  } while (0)

#elif SPH_CUBEHASH_UNROLL == 4

#define SIXTEEN_ROUNDS   do { \
    int j; \
    for (j = 0; j < 4; j ++) { \
      ROUND_EVEN; \
      ROUND_ODD; \
      ROUND_EVEN; \
      ROUND_ODD; \
    } \
  } while (0)

#elif SPH_CUBEHASH_UNROLL == 8

#define SIXTEEN_ROUNDS   do { \
    int j; \
    for (j = 0; j < 2; j ++) { \
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

// opencl versions
#define ROTL64(x,n) rotate(x,(ulong)n)
#define ROTR64(x,n) rotate(x,(ulong)(64-n))
#define SWAP32(x) as_ulong(as_uint2(x).s10)
//#define ROTL64(x,n) SPH_ROTL64(x, n)
//#define ROTR64(x,n) SPH_ROTR64(x, n)
//#define SWAP32(x) sph_bswap32(x)

/*One Round of the Blake2b's compression function*/
#define round_lyra(s)  \
 do { \
        s[0].x += s[1].x; s[3].x ^= s[0].x; s[3].x = ROTR64(s[3].x, 32); \
        s[2].x += s[3].x; s[1].x ^= s[2].x; s[1].x = ROTR64(s[1].x, 24); \
        s[0].x += s[1].x; s[3].x ^= s[0].x; s[3].x = ROTR64(s[3].x, 16); \
        s[2].x += s[3].x; s[1].x ^= s[2].x; s[1].x = ROTR64(s[1].x, 63); \
        s[0].y += s[1].y; s[3].y ^= s[0].y; s[3].y = ROTR64(s[3].y, 32); \
        s[2].y += s[3].y; s[1].y ^= s[2].y; s[1].y = ROTR64(s[1].y, 24); \
        s[0].y += s[1].y; s[3].y ^= s[0].y; s[3].y = ROTR64(s[3].y, 16); \
        s[2].y += s[3].y; s[1].y ^= s[2].y; s[1].y = ROTR64(s[1].y, 63); \
        s[0].z += s[1].z; s[3].z ^= s[0].z; s[3].z = ROTR64(s[3].z, 32); \
        s[2].z += s[3].z; s[1].z ^= s[2].z; s[1].z = ROTR64(s[1].z, 24); \
        s[0].z += s[1].z; s[3].z ^= s[0].z; s[3].z = ROTR64(s[3].z, 16); \
        s[2].z += s[3].z; s[1].z ^= s[2].z; s[1].z = ROTR64(s[1].z, 63); \
        s[0].w += s[1].w; s[3].w ^= s[0].w; s[3].w = ROTR64(s[3].w, 32); \
        s[2].w += s[3].w; s[1].w ^= s[2].w; s[1].w = ROTR64(s[1].w, 24); \
        s[0].w += s[1].w; s[3].w ^= s[0].w; s[3].w = ROTR64(s[3].w, 16); \
        s[2].w += s[3].w; s[1].w ^= s[2].w; s[1].w = ROTR64(s[1].w, 63); \
        s[0].x += s[1].y; s[3].w ^= s[0].x; s[3].w = ROTR64(s[3].w, 32); \
        s[2].z += s[3].w; s[1].y ^= s[2].z; s[1].y = ROTR64(s[1].y, 24); \
        s[0].x += s[1].y; s[3].w ^= s[0].x; s[3].w = ROTR64(s[3].w, 16); \
        s[2].z += s[3].w; s[1].y ^= s[2].z; s[1].y = ROTR64(s[1].y, 63); \
        s[0].y += s[1].z; s[3].x ^= s[0].y; s[3].x = ROTR64(s[3].x, 32); \
        s[2].w += s[3].x; s[1].z ^= s[2].w; s[1].z = ROTR64(s[1].z, 24); \
        s[0].y += s[1].z; s[3].x ^= s[0].y; s[3].x = ROTR64(s[3].x, 16); \
        s[2].w += s[3].x; s[1].z ^= s[2].w; s[1].z = ROTR64(s[1].z, 63); \
        s[0].z += s[1].w; s[3].y ^= s[0].z; s[3].y = ROTR64(s[3].y, 32); \
        s[2].x += s[3].y; s[1].w ^= s[2].x; s[1].w = ROTR64(s[1].w, 24); \
        s[0].z += s[1].w; s[3].y ^= s[0].z; s[3].y = ROTR64(s[3].y, 16); \
        s[2].x += s[3].y; s[1].w ^= s[2].x; s[1].w = ROTR64(s[1].w, 63); \
        s[0].w += s[1].x; s[3].z ^= s[0].w; s[3].z = ROTR64(s[3].z, 32); \
        s[2].y += s[3].z; s[1].x ^= s[2].y; s[1].x = ROTR64(s[1].x, 24); \
        s[0].w += s[1].x; s[3].z ^= s[0].w; s[3].z = ROTR64(s[3].z, 16); \
        s[2].y += s[3].z; s[1].x ^= s[2].y; s[1].x = ROTR64(s[1].x, 63); \
 } while(0)

#define SPH_ULONG4(a, b, c, d) (ulong4)(a, b, c, d)
#define MIX_WORDS 16

__attribute__((reqd_work_group_size(WORKSIZE, 1, 1)))
__kernel void search(
	__global volatile uint* restrict g_output,
	__constant uint const* g_header,
	__global uint16* const g_dag,
	__global uchar* g_lyre_nodes,
	const ulong DAG_ITEM_COUNT,
	const uint height,
	const ulong target
	)
{
	const uint gid = get_global_id(0); // i.e. nonce
	const uint hash_output_idx = gid;// - get_global_offset(0);
	__global ulong4 *DMatrix = (__global ulong4 *)(g_lyre_nodes + (1536 * (hash_output_idx % MAX_GLOBAL_THREADS)));

	uint saved_height = height;
	ulong saved_target = target;

	// set block
	uint block[20];
	block[0] = g_header[0];
	block[1] = g_header[1];
	block[2] = g_header[2];
	block[3] = g_header[3];
	block[4] = g_header[4];
	block[5] = g_header[5];
	block[6] = g_header[6];
	block[7] = g_header[7];

/////inlinefunc
	//blake32

	BLAKE256_COMPRESS32_STATE;

#ifndef PRECALC_BLAKE
	//T0 = SPH_T32(T0 + 512);
	// Input == input words, already byteswapped
	BLAKE256_COMPRESS_BEGIN_DIRECT(SPH_C32(0x6a09e667), SPH_C32(0xbb67ae85), SPH_C32(0x3c6ef372), SPH_C32(0xa54ff53a),  SPH_C32(0x510e527f), SPH_C32(0x9b05688c), SPH_C32(0x1f83d9ab), SPH_C32(0x5be0cd19),
		512, 0, (block[0]),(block[1]),(block[2]),(block[3]),(block[4]),(block[5]),(block[6]),(block[7]),(g_header[8]),(g_header[9]),(g_header[10]),(g_header[11]),(g_header[12]),(g_header[13]),(g_header[14]),(g_header[15]));
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
		                                block[0], block[1], block[2], block[3], block[4], block[5], block[6], block[7]);
#endif

	// NOTE: At this stage, block[0...7] contains the previous state of the blake hash so we can continue on similarly to before

	BLAKE256_COMPRESS_BEGIN_DIRECT(block[0], block[1], block[2], block[3], block[4], block[5], block[6], block[7], 640, 0, 
		                          (g_header[16]),(g_header[17]),(g_header[18]),(gid),2147483648,0,0,0,0,0,0,0,0,1,0,640);
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

	uint hashedHeader[50];

	BLAKE256_COMPRESS_END_DIRECT_NOSWAP(block[0], block[1], block[2], block[3], block[4], block[5], block[6], block[7], 
		                         hashedHeader[0], hashedHeader[1], hashedHeader[2], hashedHeader[3], hashedHeader[4], hashedHeader[5], hashedHeader[6], hashedHeader[7]);

	hashedHeader[0] = sph_bswap32(hashedHeader[0]);
	hashedHeader[1] = sph_bswap32(hashedHeader[1]);
	hashedHeader[2] = sph_bswap32(hashedHeader[2]);
	hashedHeader[3] = sph_bswap32(hashedHeader[3]);
	hashedHeader[4] = sph_bswap32(hashedHeader[4]);
	hashedHeader[5] = sph_bswap32(hashedHeader[5]);
	hashedHeader[6] = sph_bswap32(hashedHeader[6]);
	hashedHeader[7] = sph_bswap32(hashedHeader[7]);


	//keccak32
	ulong* keccak_gpu_state = (ulong*)hashedHeader;
	keccak_gpu_state[5]=keccak_gpu_state[6]=keccak_gpu_state[7]=keccak_gpu_state[8]=keccak_gpu_state[9]=keccak_gpu_state[10]=keccak_gpu_state[11]=keccak_gpu_state[12]=keccak_gpu_state[13]=keccak_gpu_state[14]=keccak_gpu_state[15]=keccak_gpu_state[16]=keccak_gpu_state[17]=keccak_gpu_state[18]=keccak_gpu_state[19]=keccak_gpu_state[20]=keccak_gpu_state[21]=keccak_gpu_state[22]=keccak_gpu_state[23]=keccak_gpu_state[24]=0;
	keccak_gpu_state[4] = 0x0000000000000001;
	keccak_gpu_state[16] = 0x8000000000000000;
	ulong t[5], u[5], v, w;
	for (size_t i = 0; i < 24; i++) {
			t[0] = keccak_gpu_state[0] ^ keccak_gpu_state[5] ^ keccak_gpu_state[10] ^ keccak_gpu_state[15] ^ keccak_gpu_state[20];
			t[1] = keccak_gpu_state[1] ^ keccak_gpu_state[6] ^ keccak_gpu_state[11] ^ keccak_gpu_state[16] ^ keccak_gpu_state[21];
			t[2] = keccak_gpu_state[2] ^ keccak_gpu_state[7] ^ keccak_gpu_state[12] ^ keccak_gpu_state[17] ^ keccak_gpu_state[22];
			t[3] = keccak_gpu_state[3] ^ keccak_gpu_state[8] ^ keccak_gpu_state[13] ^ keccak_gpu_state[18] ^ keccak_gpu_state[23];
			t[4] = keccak_gpu_state[4] ^ keccak_gpu_state[9] ^ keccak_gpu_state[14] ^ keccak_gpu_state[19] ^ keccak_gpu_state[24];
			u[0] = t[4] ^ SPH_ROTL64(t[1], 1);
			u[1] = t[0] ^ SPH_ROTL64(t[2], 1);
			u[2] = t[1] ^ SPH_ROTL64(t[3], 1);
			u[3] = t[2] ^ SPH_ROTL64(t[4], 1);
			u[4] = t[3] ^ SPH_ROTL64(t[0], 1);
			keccak_gpu_state[0] ^= u[0]; keccak_gpu_state[5] ^= u[0]; keccak_gpu_state[10] ^= u[0]; keccak_gpu_state[15] ^= u[0]; keccak_gpu_state[20] ^= u[0];
			keccak_gpu_state[1] ^= u[1]; keccak_gpu_state[6] ^= u[1]; keccak_gpu_state[11] ^= u[1]; keccak_gpu_state[16] ^= u[1]; keccak_gpu_state[21] ^= u[1];
			keccak_gpu_state[2] ^= u[2]; keccak_gpu_state[7] ^= u[2]; keccak_gpu_state[12] ^= u[2]; keccak_gpu_state[17] ^= u[2]; keccak_gpu_state[22] ^= u[2];
			keccak_gpu_state[3] ^= u[3]; keccak_gpu_state[8] ^= u[3]; keccak_gpu_state[13] ^= u[3]; keccak_gpu_state[18] ^= u[3]; keccak_gpu_state[23] ^= u[3];
			keccak_gpu_state[4] ^= u[4]; keccak_gpu_state[9] ^= u[4]; keccak_gpu_state[14] ^= u[4]; keccak_gpu_state[19] ^= u[4]; keccak_gpu_state[24] ^= u[4];
			v = keccak_gpu_state[1];
			keccak_gpu_state[1] = SPH_ROTL64(keccak_gpu_state[6], 44);
			keccak_gpu_state[6] = SPH_ROTL64(keccak_gpu_state[9], 20);
			keccak_gpu_state[9] = SPH_ROTL64(keccak_gpu_state[22], 61);
			keccak_gpu_state[22] = SPH_ROTL64(keccak_gpu_state[14], 39);
			keccak_gpu_state[14] = SPH_ROTL64(keccak_gpu_state[20], 18);
			keccak_gpu_state[20] = SPH_ROTL64(keccak_gpu_state[2], 62);
			keccak_gpu_state[2] = SPH_ROTL64(keccak_gpu_state[12], 43);
			keccak_gpu_state[12] = SPH_ROTL64(keccak_gpu_state[13], 25);
			keccak_gpu_state[13] = SPH_ROTL64(keccak_gpu_state[19], 8);
			keccak_gpu_state[19] = SPH_ROTL64(keccak_gpu_state[23], 56);
			keccak_gpu_state[23] = SPH_ROTL64(keccak_gpu_state[15], 41);
			keccak_gpu_state[15] = SPH_ROTL64(keccak_gpu_state[4], 27);
			keccak_gpu_state[4] = SPH_ROTL64(keccak_gpu_state[24], 14);
			keccak_gpu_state[24] = SPH_ROTL64(keccak_gpu_state[21], 2);
			keccak_gpu_state[21] = SPH_ROTL64(keccak_gpu_state[8], 55);
			keccak_gpu_state[8] = SPH_ROTL64(keccak_gpu_state[16], 45);
			keccak_gpu_state[16] = SPH_ROTL64(keccak_gpu_state[5], 36);
			keccak_gpu_state[5] = SPH_ROTL64(keccak_gpu_state[3], 28);
			keccak_gpu_state[3] = SPH_ROTL64(keccak_gpu_state[18], 21);
			keccak_gpu_state[18] = SPH_ROTL64(keccak_gpu_state[17], 15);
			keccak_gpu_state[17] = SPH_ROTL64(keccak_gpu_state[11], 10);
			keccak_gpu_state[11] = SPH_ROTL64(keccak_gpu_state[7], 6);
			keccak_gpu_state[7] = SPH_ROTL64(keccak_gpu_state[10], 3);
			keccak_gpu_state[10] = SPH_ROTL64(v, 1);
			v = keccak_gpu_state[0]; w = keccak_gpu_state[1]; keccak_gpu_state[0] ^= (~w) & keccak_gpu_state[2]; keccak_gpu_state[1] ^= (~keccak_gpu_state[2]) & keccak_gpu_state[3]; keccak_gpu_state[2] ^= (~keccak_gpu_state[3]) & keccak_gpu_state[4]; keccak_gpu_state[3] ^= (~keccak_gpu_state[4]) & v; keccak_gpu_state[4] ^= (~v) & w;
			v = keccak_gpu_state[5]; w = keccak_gpu_state[6]; keccak_gpu_state[5] ^= (~w) & keccak_gpu_state[7]; keccak_gpu_state[6] ^= (~keccak_gpu_state[7]) & keccak_gpu_state[8]; keccak_gpu_state[7] ^= (~keccak_gpu_state[8]) & keccak_gpu_state[9]; keccak_gpu_state[8] ^= (~keccak_gpu_state[9]) & v; keccak_gpu_state[9] ^= (~v) & w;
			v = keccak_gpu_state[10]; w = keccak_gpu_state[11]; keccak_gpu_state[10] ^= (~w) & keccak_gpu_state[12]; keccak_gpu_state[11] ^= (~keccak_gpu_state[12]) & keccak_gpu_state[13]; keccak_gpu_state[12] ^= (~keccak_gpu_state[13]) & keccak_gpu_state[14]; keccak_gpu_state[13] ^= (~keccak_gpu_state[14]) & v; keccak_gpu_state[14] ^= (~v) & w;
			v = keccak_gpu_state[15]; w = keccak_gpu_state[16]; keccak_gpu_state[15] ^= (~w) & keccak_gpu_state[17]; keccak_gpu_state[16] ^= (~keccak_gpu_state[17]) & keccak_gpu_state[18]; keccak_gpu_state[17] ^= (~keccak_gpu_state[18]) & keccak_gpu_state[19]; keccak_gpu_state[18] ^= (~keccak_gpu_state[19]) & v; keccak_gpu_state[19] ^= (~v) & w;
			v = keccak_gpu_state[20]; w = keccak_gpu_state[21]; keccak_gpu_state[20] ^= (~w) & keccak_gpu_state[22]; keccak_gpu_state[21] ^= (~keccak_gpu_state[22]) & keccak_gpu_state[23]; keccak_gpu_state[22] ^= (~keccak_gpu_state[23]) & keccak_gpu_state[24]; keccak_gpu_state[23] ^= (~keccak_gpu_state[24]) & v; keccak_gpu_state[24] ^= (~v) & w;
			keccak_gpu_state[0] ^= RC[i];
	}
	uint* hashB = (uint*)keccak_gpu_state;
        //cubehash32
	uint x0 = 0xEA2BD4B4; uint x1 = 0xCCD6F29F; uint x2 = 0x63117E71;
	uint x3 = 0x35481EAE; uint x4 = 0x22512D5B; uint x5 = 0xE5D94E63;
	uint x6 = 0x7E624131; uint x7 = 0xF4CC12BE; uint x8 = 0xC2D0B696;
	uint x9 = 0x42AF2070; uint xa = 0xD0720C35; uint xb = 0x3361DA8C;
	uint xc = 0x28CCECA4; uint xd = 0x8EF8AD83; uint xe = 0x4680AC00;
	uint xf = 0x40E5FBAB;
	uint xg = 0xD89041C3; uint xh = 0x6107FBD5;
	uint xi = 0x6C859D41; uint xj = 0xF0B26679; uint xk = 0x09392549;
	uint xl = 0x5FA25603; uint xm = 0x65C892FD; uint xn = 0x93CB6285;
	uint xo = 0x2AF2B5AE; uint xp = 0x9E4B4E60; uint xq = 0x774ABFDD;
	uint xr = 0x85254725; uint xs = 0x15815AEB; uint xt = 0x4AB6AAD6;
	uint xu = 0x9CDAF8AF; uint xv = 0xD6032C0A;
	x0 ^= (hashB[0]);
	x1 ^= (hashB[1]);
	x2 ^= (hashB[2]);
	x3 ^= (hashB[3]);
	x4 ^= (hashB[4]);
	x5 ^= (hashB[5]);
	x6 ^= (hashB[6]);
	x7 ^= (hashB[7]);
	SIXTEEN_ROUNDS;
	x0 ^= 0x80;
	SIXTEEN_ROUNDS;
	xv ^= 0x01;
	for (int i = 0; i < 10; ++i) SIXTEEN_ROUNDS;
	hashedHeader[0] = x0;
	hashedHeader[1] = x1;
	hashedHeader[2] = x2;
	hashedHeader[3] = x3;
	hashedHeader[4] = x4;
	hashedHeader[5] = x5;
	hashedHeader[6] = x6;
	hashedHeader[7] = x7;
    //lyra2 start
	ulong4* state = (ulong4*)hashedHeader;
	state[1] = state[0]; state[2] = SPH_ULONG4(0x6a09e667f3bcc908UL, 0xbb67ae8584caa73bUL, 0x3c6ef372fe94f82bUL, 0xa54ff53a5f1d36f1UL); state[3] = SPH_ULONG4(0x510e527fade682d1UL, 0x9b05688c2b3e6c1fUL, 0x1f83d9abfb41bd6bUL, 0x5be0cd19137e2179UL);	for (int i = 0; i<12; i++) round_lyra(state); state[0] ^= SPH_ULONG4(0x20,0x20,0x20,0x01); state[1] ^= SPH_ULONG4(0x04,0x04,0x80,0x0100000000000000); for (int i = 0; i<12; i++) round_lyra(state); uint ps1 = (3 * 3); for (int i = 0; i < 4; i++) { uint s1 = ps1 - 3 * i; for (int j = 0; j < 3; j++) (DMatrix)[j+s1] = state[j]; round_lyra(state); }	
    // squashed reduceduplexf - reduceDuplexf(state,DMatrix)
	ulong4 state1[3]; ps1 = 0; uint ps2 = (3 * 3 + 3 * 4); for (int i = 0; i < 4; i++) { uint s1 = ps1 + i*3; uint s2 = ps2 - i*3; for (int j = 0; j < 3; j++) state1[j] = (DMatrix)[j + s1]; for (int j = 0; j < 3; j++) state[j] ^= state1[j]; round_lyra(state); for (int j = 0; j < 3; j++) state1[j] ^= state[j]; for (int j = 0; j < 3; j++) (DMatrix)[j + s2] = state1[j]; }	
	// squashed reduceduplexrowsetupf (1,0,2..) - reduceDuplexRowSetupf(uint rowIn, uint rowInOut, uint rowOut, ulong4 *state,  __global ulong4* DMatrix)
	ulong4 state2[3]; ps1 = (3 * 4 * 1); ps2 = (3 * 4 * 0); uint ps3 = (3 * 3 + 3 * 4 * 2); for (int i = 0; i < 4; i++) { uint s1 = ps1 + i*3; uint s2 = ps2 + i*3; uint s3 = ps3 - i*3;	for (int j = 0; j < 3; j++) state1[j] = (DMatrix)[j + s1]; for (int j = 0; j < 3; j++) state2[j] = (DMatrix)[j + s2]; for (int j = 0; j < 3; j++) { ulong4 tmp = state1[j] + state2[j]; state[j] ^= tmp; } round_lyra(state); for (int j = 0; j < 3; j++) { state1[j] ^= state[j]; (DMatrix)[j + s3] = state1[j]; } ((ulong*)state2)[0] ^= ((ulong*)state)[11];	for (int j = 0; j < 11; j++) ((ulong*)state2)[j + 1] ^= ((ulong*)state)[j];	for (int j = 0; j < 3; j++) (DMatrix)[j + s2] = state2[j]; }
	// squashed reduceduplexrowsetupf (2,1,3..) - reduceDuplexRowSetupf(uint rowIn, uint rowInOut, uint rowOut, ulong4 *state,  __global ulong4* DMatrix)
	ps1 = (3 * 4 * 2); ps2 = (3 * 4 * 1); ps3 = (3 * 3 + 3 * 4 * 3); for (int i = 0; i < 4; i++) { uint s1 = ps1 + i*3; uint s2 = ps2 + i*3; uint s3 = ps3 - i*3;	for (int j = 0; j < 3; j++) state1[j] = (DMatrix)[j + s1]; for (int j = 0; j < 3; j++) state2[j] = (DMatrix)[j + s2]; for (int j = 0; j < 3; j++) { ulong4 tmp = state1[j] + state2[j]; state[j] ^= tmp; } round_lyra(state); for (int j = 0; j < 3; j++) { state1[j] ^= state[j]; (DMatrix)[j + s3] = state1[j]; } ((ulong*)state2)[0] ^= ((ulong*)state)[11];	for (int j = 0; j < 11; j++) ((ulong*)state2)[j + 1] ^= ((ulong*)state)[j];	for (int j = 0; j < 3; j++) (DMatrix)[j + s2] = state2[j]; }
    // last loops of lyra2
	uint rowa; uint prev = 3; for (uint h = 0; h<4; h++) { rowa = state[0].x & 3; ulong4 state1[3], state2[3]; uint ps1 = (3 * 4 * prev);   uint ps2 = (3 * 4 * rowa); uint ps3 = (3 * 4 * h); for (int i = 0; i < 4; i++) { uint s1 = ps1 + i*3; uint s2 = ps2 + i*3; uint s3 = ps3 + i*3; for (int j = 0; j < 3; j++) state1[j] = (DMatrix)[j + s1]; for (int j = 0; j < 3; j++) state2[j] = (DMatrix)[j + s2]; for (int j = 0; j < 3; j++) state1[j] += state2[j]; for (int j = 0; j < 3; j++) state[j] ^= state1[j]; round_lyra(state); ((ulong*)state2)[0] ^= ((ulong*)state)[11]; for (int j = 0; j < 11; j++) ((ulong*)state2)[j + 1] ^= ((ulong*)state)[j];   if (rowa != h) { for (int j = 0; j < 3; j++) (DMatrix)[j + s2] = state2[j]; for (int j = 0; j < 3; j++) (DMatrix)[j + s3] ^= state[j];  } else { for (int j = 0; j < 3; j++) state2[j] ^= state[j]; for (int j = 0; j < 3; j++) (DMatrix)[j + s2] = state2[j]; }} prev = h; } uint shift = (3 * 4 * rowa);	for (int j = 0; j < 3; j++) state[j] ^= (DMatrix)[j+shift]; for (int i = 0; i < 12; i++) round_lyra(state);	
	hashB = (uint*)state;
        //keccak32
        ulong* in_dwords = (ulong*)hashB;
        ulong h[9]; ulong dt0,dt1,dt2,dt3;ulong p0,p1,p2,p3,p4,p5,p6,p7;h[8] = skein_ks_parity;
        for (int i = 0; i<8; i++) { h[i] = SKEIN_IV512_256[i]; h[8] ^= h[i]; }  t[0]=t12[0]; t[1]=t12[1]; t[2]=t12[2];
        dt0= (in_dwords[0]); dt1= (in_dwords[1]); dt2= (in_dwords[2]); dt3= (in_dwords[3]);
        p0 = h[0] + dt0;p1 = h[1] + dt1;p2 = h[2] + dt2;p3 = h[3] + dt3;p4 = h[4];p5 = h[5] + t[0];p6 = h[6] + t[1];p7 = h[7];
        for (int i = 1; i<19; i+=2) {Round_8_512(p0,p1,p2,p3,p4,p5,p6,p7,i);}
        p0 ^= dt0; p1 ^= dt1; p2 ^= dt2; p3 ^= dt3;
        h[0] = p0;h[1] = p1;h[2] = p2;h[3] = p3;h[4] = p4;h[5] = p5;h[6] = p6;h[7] = p7;h[8] = skein_ks_parity;
        for (int i = 0; i<8; i++) { h[8] ^= h[i]; } t[0] = t12[3];t[1] = t12[4];t[2] = t12[5];p5 += t[0];p6 += t[1];
        for (int i = 1; i<19; i+=2) { Round_8_512(p0, p1, p2, p3, p4, p5, p6, p7, i); }
        ulong out_dwords[50]; 
        out_dwords[0] = (p0);out_dwords[1] = (p1);out_dwords[2] = (p2); out_dwords[3] = (p3);
        hashB = (uint*)out_dwords;
    //cubehash32
	x0 = 0xEA2BD4B4; x1 = 0xCCD6F29F; x2 = 0x63117E71;
	x3 = 0x35481EAE; x4 = 0x22512D5B; x5 = 0xE5D94E63;
	x6 = 0x7E624131; x7 = 0xF4CC12BE; x8 = 0xC2D0B696;
	x9 = 0x42AF2070; xa = 0xD0720C35; xb = 0x3361DA8C;
	xc = 0x28CCECA4; xd = 0x8EF8AD83; xe = 0x4680AC00;
	xf = 0x40E5FBAB;
	xg = 0xD89041C3; xh = 0x6107FBD5;
	xi = 0x6C859D41; xj = 0xF0B26679; xk = 0x09392549;
	xl = 0x5FA25603; xm = 0x65C892FD; xn = 0x93CB6285;
	xo = 0x2AF2B5AE; xp = 0x9E4B4E60; xq = 0x774ABFDD;
	xr = 0x85254725; xs = 0x15815AEB; xt = 0x4AB6AAD6;
	xu = 0x9CDAF8AF; xv = 0xD6032C0A;
	x0 ^= (hashB[0]);
	x1 ^= (hashB[1]);
	x2 ^= (hashB[2]);
	x3 ^= (hashB[3]);
	x4 ^= (hashB[4]);
	x5 ^= (hashB[5]);
	x6 ^= (hashB[6]);
	x7 ^= (hashB[7]);
	SIXTEEN_ROUNDS;
	x0 ^= 0x80;
	SIXTEEN_ROUNDS;
	xv ^= 0x01;
	for (int i = 0; i < 10; ++i) SIXTEEN_ROUNDS;
    //bmw32
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
	uint final_s[16] = {
			0xaaaaaaa0, 0xaaaaaaa1, 0xaaaaaaa2,
			0xaaaaaaa3, 0xaaaaaaa4, 0xaaaaaaa5,
			0xaaaaaaa6, 0xaaaaaaa7, 0xaaaaaaa8,
			0xaaaaaaa9, 0xaaaaaaaa, 0xaaaaaaab,
			0xaaaaaaac, 0xaaaaaaad, 0xaaaaaaae,
			0xaaaaaaaf
	};
	uint message[16]; 
        message[0] = x0; message[1] = x1; message[2] = x2; message[3] = x3; message[4] = x4; 
        message[5] = x5; message[6] = x6; message[7] = x7; message[9] = 0; message[10] = 0;
        message[11] = 0; message[12] = 0; message[13] = 0; message[8]= 0x80; message[14]=0x100; message[15]=0;
        //compression256 message_dh_start
	uint XL32,XH32,Q[32];
	Q[0]=ss0((message[5]^dh[5])-(message[7]^dh[7])+(message[10]^dh[10])+(message[13]^dh[13])+(message[14]^dh[14]))+dh[1];
	Q[1]=ss1((message[6]^dh[6])-(message[8]^dh[8])+(message[11]^dh[11])+(message[14]^dh[14])-(message[15]^dh[15]))+dh[2];
	Q[2]=ss2((message[0]^dh[0])+(message[7]^dh[7])+(message[9]^dh[9])-(message[12]^dh[12])+(message[15]^dh[15]))+dh[3];
	Q[3]=ss3((message[0]^dh[0])-(message[1]^dh[1])+(message[8]^dh[8])-(message[10]^dh[10])+(message[13]^dh[13]))+dh[4];
	Q[4]=ss4((message[1]^dh[1])+(message[2]^dh[2])+(message[9]^dh[9])-(message[11]^dh[11])-(message[14]^dh[14]))+dh[5];
	Q[5]=ss0((message[3]^dh[3])-(message[2]^dh[2])+(message[10]^dh[10])-(message[12]^dh[12])+(message[15]^dh[15]))+dh[6];
	Q[6]=ss1((message[4]^dh[4])-(message[0]^dh[0])-(message[3]^dh[3])-(message[11]^dh[11])+(message[13]^dh[13]))+dh[7];
	Q[7]=ss2((message[1]^dh[1])-(message[4]^dh[4])-(message[5]^dh[5])-(message[12]^dh[12])-(message[14]^dh[14]))+dh[8];
	Q[8]=ss3((message[2]^dh[2])-(message[5]^dh[5])-(message[6]^dh[6])+(message[13]^dh[13])-(message[15]^dh[15]))+dh[9];
	Q[9]=ss4((message[0]^dh[0])-(message[3]^dh[3])+(message[6]^dh[6])-(message[7]^dh[7])+(message[14]^dh[14]))+dh[10];
	Q[10]=ss0((message[8]^dh[8])-(message[1]^dh[1])-(message[4]^dh[4])-(message[7]^dh[7])+(message[15]^dh[15]))+dh[11];
	Q[11]=ss1((message[8]^dh[8])-(message[0]^dh[0])-(message[2]^dh[2])-(message[5]^dh[5])+(message[9]^dh[9]))+dh[12];
	Q[12]=ss2((message[1]^dh[1])+(message[3]^dh[3])-(message[6]^dh[6])-(message[9]^dh[9])+(message[10]^dh[10]))+dh[13];
	Q[13]=ss3((message[2]^dh[2])+(message[4]^dh[4])+(message[7]^dh[7])+(message[10]^dh[10])+(message[11]^dh[11]))+dh[14];
	Q[14]=ss4((message[3]^dh[3])-(message[5]^dh[5])+(message[8]^dh[8])-(message[11]^dh[11])-(message[12]^dh[12]))+dh[15];
	Q[15]=ss0((message[12]^dh[12])-(message[4]^dh[4])-(message[6]^dh[6])-(message[9]^dh[9])+(message[13]^dh[13]))+dh[0];
	Q[16]=expand32_1(16,message,dh,Q);
	Q[17]=expand32_1(17,message,dh,Q);
	Q[18]=expand32_2(18,message,dh,Q);
	Q[19]=expand32_2(19,message,dh,Q);
	Q[20]=expand32_2(20,message,dh,Q);
	Q[21]=expand32_2(21,message,dh,Q);
	Q[22]=expand32_2(22,message,dh,Q);
	Q[23]=expand32_2(23,message,dh,Q);
	Q[24]=expand32_2(24,message,dh,Q);
	Q[25]=expand32_2(25,message,dh,Q);
	Q[26]=expand32_2(26,message,dh,Q);
	Q[27]=expand32_2(27,message,dh,Q);
	Q[28]=expand32_2(28,message,dh,Q);
	Q[29]=expand32_2(29,message,dh,Q);
	Q[30]=expand32_2(30,message,dh,Q);
	Q[31]=expand32_2(31,message,dh,Q);
	XL32=Q[16]^Q[17]^Q[18]^Q[19]^Q[20]^Q[21]^Q[22]^Q[23];
	XH32=XL32^Q[24]^Q[25]^Q[26]^Q[27]^Q[28]^Q[29]^Q[30]^Q[31];
	dh[0]=(shl(XH32,5)^shr(Q[16],5)^message[0])+(XL32^Q[24]^Q[0]);
	dh[1]=(shr(XH32,7)^shl(Q[17],8)^message[1])+(XL32^Q[25]^Q[1]);
	dh[2]=(shr(XH32,5)^shl(Q[18],5)^message[2])+(XL32^Q[26]^Q[2]);
	dh[3]=(shr(XH32,1)^shl(Q[19],5)^message[3])+(XL32^Q[27]^Q[3]);
	dh[4]=(shr(XH32,3)^Q[20]^message[4])+(XL32^Q[28]^Q[4]);
	dh[5]=(shl(XH32,6)^shr(Q[21],6)^message[5])+(XL32^Q[29]^Q[5]);
	dh[6]=(shr(XH32,4)^shl(Q[22],6)^message[6])+(XL32^Q[30]^Q[6]);
	dh[7]=(shr(XH32,11)^shl(Q[23],2)^message[7])+(XL32^Q[31]^Q[7]);
	dh[8]=SPH_ROTL32(dh[4],9)+(XH32^Q[24]^message[8])+(shl(XL32,8)^Q[23]^Q[8]);
	dh[9]=SPH_ROTL32(dh[5],10)+(XH32^Q[25]^message[9])+(shr(XL32,6)^Q[16]^Q[9]);
	dh[10]=SPH_ROTL32(dh[6],11)+(XH32^Q[26]^message[10])+(shl(XL32,6)^Q[17]^Q[10]);
	dh[11]=SPH_ROTL32(dh[7],12)+(XH32^Q[27]^message[11])+(shl(XL32,4)^Q[18]^Q[11]);
	dh[12]=SPH_ROTL32(dh[0],13)+(XH32^Q[28]^message[12])+(shr(XL32,3)^Q[19]^Q[12]);
	dh[13]=SPH_ROTL32(dh[1],14)+(XH32^Q[29]^message[13])+(shr(XL32,4)^Q[20]^Q[13]);
	dh[14]=SPH_ROTL32(dh[2],15)+(XH32^Q[30]^message[14])+(shr(XL32,7)^Q[21]^Q[14]);
	dh[15]=SPH_ROTL32(dh[3],16)+(XH32^Q[31]^message[15])+(shr(XL32,2)^Q[22]^Q[15]);
        //compression256 message_dh_end
	//compression256 dh_finals_start
	Q[0]=ss0((dh[5]^final_s[5])-(dh[7]^final_s[7])+(dh[10]^final_s[10])+(dh[13]^final_s[13])+(dh[14]^final_s[14]))+final_s[1];
	Q[1]=ss1((dh[6]^final_s[6])-(dh[8]^final_s[8])+(dh[11]^final_s[11])+(dh[14]^final_s[14])-(dh[15]^final_s[15]))+final_s[2];
	Q[2]=ss2((dh[0]^final_s[0])+(dh[7]^final_s[7])+(dh[9]^final_s[9])-(dh[12]^final_s[12])+(dh[15]^final_s[15]))+final_s[3];
	Q[3]=ss3((dh[0]^final_s[0])-(dh[1]^final_s[1])+(dh[8]^final_s[8])-(dh[10]^final_s[10])+(dh[13]^final_s[13]))+final_s[4];
	Q[4]=ss4((dh[1]^final_s[1])+(dh[2]^final_s[2])+(dh[9]^final_s[9])-(dh[11]^final_s[11])-(dh[14]^final_s[14]))+final_s[5];
	Q[5]=ss0((dh[3]^final_s[3])-(dh[2]^final_s[2])+(dh[10]^final_s[10])-(dh[12]^final_s[12])+(dh[15]^final_s[15]))+final_s[6];
	Q[6]=ss1((dh[4]^final_s[4])-(dh[0]^final_s[0])-(dh[3]^final_s[3])-(dh[11]^final_s[11])+(dh[13]^final_s[13]))+final_s[7];
	Q[7]=ss2((dh[1]^final_s[1])-(dh[4]^final_s[4])-(dh[5]^final_s[5])-(dh[12]^final_s[12])-(dh[14]^final_s[14]))+final_s[8];
	Q[8]=ss3((dh[2]^final_s[2])-(dh[5]^final_s[5])-(dh[6]^final_s[6])+(dh[13]^final_s[13])-(dh[15]^final_s[15]))+final_s[9];
	Q[9]=ss4((dh[0]^final_s[0])-(dh[3]^final_s[3])+(dh[6]^final_s[6])-(dh[7]^final_s[7])+(dh[14]^final_s[14]))+final_s[10];
	Q[10]=ss0((dh[8]^final_s[8])-(dh[1]^final_s[1])-(dh[4]^final_s[4])-(dh[7]^final_s[7])+(dh[15]^final_s[15]))+final_s[11];
	Q[11]=ss1((dh[8]^final_s[8])-(dh[0]^final_s[0])-(dh[2]^final_s[2])-(dh[5]^final_s[5])+(dh[9]^final_s[9]))+final_s[12];
	Q[12]=ss2((dh[1]^final_s[1])+(dh[3]^final_s[3])-(dh[6]^final_s[6])-(dh[9]^final_s[9])+(dh[10]^final_s[10]))+final_s[13];
	Q[13]=ss3((dh[2]^final_s[2])+(dh[4]^final_s[4])+(dh[7]^final_s[7])+(dh[10]^final_s[10])+(dh[11]^final_s[11]))+final_s[14];
	Q[14]=ss4((dh[3]^final_s[3])-(dh[5]^final_s[5])+(dh[8]^final_s[8])-(dh[11]^final_s[11])-(dh[12]^final_s[12]))+final_s[15];
	Q[15]=ss0((dh[12]^final_s[12])-(dh[4]^final_s[4])-(dh[6]^final_s[6])-(dh[9]^final_s[9])+(dh[13]^final_s[13]))+final_s[0];
	Q[16]=expand32_1(16,dh,final_s,Q);
	Q[17]=expand32_1(17,dh,final_s,Q);
	Q[18]=expand32_2(18,dh,final_s,Q);
	Q[19]=expand32_2(19,dh,final_s,Q);
	Q[20]=expand32_2(20,dh,final_s,Q);
	Q[21]=expand32_2(21,dh,final_s,Q);
	Q[22]=expand32_2(22,dh,final_s,Q);
	Q[23]=expand32_2(23,dh,final_s,Q);
	Q[24]=expand32_2(24,dh,final_s,Q);
	Q[25]=expand32_2(25,dh,final_s,Q);
	Q[26]=expand32_2(26,dh,final_s,Q);
	Q[27]=expand32_2(27,dh,final_s,Q);
	Q[28]=expand32_2(28,dh,final_s,Q);
	Q[29]=expand32_2(29,dh,final_s,Q);
	Q[30]=expand32_2(30,dh,final_s,Q);
	Q[31]=expand32_2(31,dh,final_s,Q);
	XL32=Q[16]^Q[17]^Q[18]^Q[19]^Q[20]^Q[21]^Q[22]^Q[23];
	XH32=XL32^Q[24]^Q[25]^Q[26]^Q[27]^Q[28]^Q[29]^Q[30]^Q[31];
	final_s[0]=(shl(XH32,5)^shr(Q[16],5)^dh[0])+(XL32^Q[24]^Q[0]);
	final_s[1]=(shr(XH32,7)^shl(Q[17],8)^dh[1])+(XL32^Q[25]^Q[1]);
	final_s[2]=(shr(XH32,5)^shl(Q[18],5)^dh[2])+(XL32^Q[26]^Q[2]);
	final_s[3]=(shr(XH32,1)^shl(Q[19],5)^dh[3])+(XL32^Q[27]^Q[3]);
	final_s[4]=(shr(XH32,3)^Q[20]^dh[4])+(XL32^Q[28]^Q[4]);
	final_s[5]=(shl(XH32,6)^shr(Q[21],6)^dh[5])+(XL32^Q[29]^Q[5]);
	final_s[6]=(shr(XH32,4)^shl(Q[22],6)^dh[6])+(XL32^Q[30]^Q[6]);
	final_s[7]=(shr(XH32,11)^shl(Q[23],2)^dh[7])+(XL32^Q[31]^Q[7]);
	final_s[8]=SPH_ROTL32(final_s[4],9)+(XH32^Q[24]^dh[8])+(shl(XL32,8)^Q[23]^Q[8]);
	final_s[9]=SPH_ROTL32(final_s[5],10)+(XH32^Q[25]^dh[9])+(shr(XL32,6)^Q[16]^Q[9]);
	final_s[10]=SPH_ROTL32(final_s[6],11)+(XH32^Q[26]^dh[10])+(shl(XL32,6)^Q[17]^Q[10]);
	final_s[11]=SPH_ROTL32(final_s[7],12)+(XH32^Q[27]^dh[11])+(shl(XL32,4)^Q[18]^Q[11]);
	final_s[12]=SPH_ROTL32(final_s[0],13)+(XH32^Q[28]^dh[12])+(shr(XL32,3)^Q[19]^Q[12]);
	final_s[13]=SPH_ROTL32(final_s[1],14)+(XH32^Q[29]^dh[13])+(shr(XL32,4)^Q[20]^Q[13]);
	final_s[14]=SPH_ROTL32(final_s[2],15)+(XH32^Q[30]^dh[14])+(shr(XL32,7)^Q[21]^Q[14]);
	final_s[15]=SPH_ROTL32(final_s[3],16)+(XH32^Q[31]^dh[15])+(shr(XL32,2)^Q[22]^Q[15]);
	//compression256 dh_finals_end
	
////inlinefunc
	const ulong mixhashes = MIX_BYTES / HASH_BYTES;    // 2
	const ulong wordhashes = MIX_BYTES / WORD_BYTES;   // 16
	MixNodes mix;                  // 64 bytes

	mix.nodes16 = (uint16)(final_s[8], final_s[9], final_s[10], final_s[11], final_s[12], final_s[13], final_s[14], final_s[15],
						   final_s[8], final_s[9], final_s[10], final_s[11], final_s[12], final_s[13], final_s[14], final_s[15]);
	for (uint i = 0; i < ACCESSES; i++) {
			uint p = fnv(i ^ final_s[8], mix.values[i % 16]) % (DAG_ITEM_COUNT / mixhashes);
			mix.nodes16 *= FNV_PRIME;
			mix.nodes16 ^= g_dag[p];
	}
	
	// cmix -> result.cmix. Also goes at end of header.
	final_s[0] = height;
	final_s[1] = fnv(fnv(fnv(mix.values[0], mix.values[0 + 1]), mix.values[0 + 2]), mix.values[0 + 3]);
	final_s[2] = fnv(fnv(fnv(mix.values[4], mix.values[4 + 1]), mix.values[4 + 2]), mix.values[4 + 3]);
	final_s[3] = fnv(fnv(fnv(mix.values[8], mix.values[8 + 1]), mix.values[8 + 2]), mix.values[8 + 3]);
	final_s[4] = fnv(fnv(fnv(mix.values[12], mix.values[12 + 1]), mix.values[12 + 2]), mix.values[12 + 3]);

/////inlinefunc

    //blake52
    BLAKE256_STATE;
	INIT_BLAKE256_STATE;
	T0 = SPH_C32(0xFFFFFE00) + 416;
	T1 = SPH_C32(0xFFFFFFFF);
	T0 = SPH_T32(T0 + 512);
	T1 = SPH_T32(T1 + 1);        
	BLAKE256_COMPRESS_BEGIN(sph_bswap32(final_s[8]),sph_bswap32(final_s[9]),sph_bswap32(final_s[10]),sph_bswap32(final_s[11]),sph_bswap32(final_s[12]),sph_bswap32(final_s[13]),sph_bswap32(final_s[14]),sph_bswap32(final_s[15]),sph_bswap32(final_s[0]),sph_bswap32(final_s[1]),sph_bswap32(final_s[2]),sph_bswap32(final_s[3]),sph_bswap32(final_s[4]),2147483649,0,416);
        #pragma unroll 14
	for (uint R = 0; R< 14; R++) {
			BLAKE256_GS_ALT(0, 4, 0x8, 0xC, 0x0);
			BLAKE256_GS_ALT(1, 5, 0x9, 0xD, 0x2);
			BLAKE256_GS_ALT(2, 6, 0xA, 0xE, 0x4);
			BLAKE256_GS_ALT(3, 7, 0xB, 0xF, 0x6);
			BLAKE256_GS_ALT(0, 5, 0xA, 0xF, 0x8);
			BLAKE256_GS_ALT(1, 6, 0xB, 0xC, 0xA);
			BLAKE256_GS_ALT(2, 7, 0x8, 0xD, 0xC);
			BLAKE256_GS_ALT(3, 4, 0x9, 0xE, 0xE);
	}
	BLAKE256_COMPRESS_END;
	hashedHeader[0] = sph_bswap32(H0);
	hashedHeader[1] = sph_bswap32(H1);
	hashedHeader[2] = sph_bswap32(H2);
	hashedHeader[3] = sph_bswap32(H3);
	hashedHeader[4] = sph_bswap32(H4);
	hashedHeader[5] = sph_bswap32(H5);
	hashedHeader[6] = sph_bswap32(H6);
	hashedHeader[7] = sph_bswap32(H7);
	//keccak32
	keccak_gpu_state = (ulong*)hashedHeader;
	keccak_gpu_state[5]=keccak_gpu_state[6]=keccak_gpu_state[7]=keccak_gpu_state[8]=keccak_gpu_state[9]=keccak_gpu_state[10]=keccak_gpu_state[11]=keccak_gpu_state[12]=keccak_gpu_state[13]=keccak_gpu_state[14]=keccak_gpu_state[15]=keccak_gpu_state[16]=keccak_gpu_state[17]=keccak_gpu_state[18]=keccak_gpu_state[19]=keccak_gpu_state[20]=keccak_gpu_state[21]=keccak_gpu_state[22]=keccak_gpu_state[23]=keccak_gpu_state[24]=0;
	keccak_gpu_state[4] = 0x0000000000000001;
	keccak_gpu_state[16] = 0x8000000000000000;
	for (size_t i = 0; i < 24; i++) {
			t[0] = keccak_gpu_state[0] ^ keccak_gpu_state[5] ^ keccak_gpu_state[10] ^ keccak_gpu_state[15] ^ keccak_gpu_state[20];
			t[1] = keccak_gpu_state[1] ^ keccak_gpu_state[6] ^ keccak_gpu_state[11] ^ keccak_gpu_state[16] ^ keccak_gpu_state[21];
			t[2] = keccak_gpu_state[2] ^ keccak_gpu_state[7] ^ keccak_gpu_state[12] ^ keccak_gpu_state[17] ^ keccak_gpu_state[22];
			t[3] = keccak_gpu_state[3] ^ keccak_gpu_state[8] ^ keccak_gpu_state[13] ^ keccak_gpu_state[18] ^ keccak_gpu_state[23];
			t[4] = keccak_gpu_state[4] ^ keccak_gpu_state[9] ^ keccak_gpu_state[14] ^ keccak_gpu_state[19] ^ keccak_gpu_state[24];
			u[0] = t[4] ^ SPH_ROTL64(t[1], 1);
			u[1] = t[0] ^ SPH_ROTL64(t[2], 1);
			u[2] = t[1] ^ SPH_ROTL64(t[3], 1);
			u[3] = t[2] ^ SPH_ROTL64(t[4], 1);
			u[4] = t[3] ^ SPH_ROTL64(t[0], 1);
			keccak_gpu_state[0] ^= u[0]; keccak_gpu_state[5] ^= u[0]; keccak_gpu_state[10] ^= u[0]; keccak_gpu_state[15] ^= u[0]; keccak_gpu_state[20] ^= u[0];
			keccak_gpu_state[1] ^= u[1]; keccak_gpu_state[6] ^= u[1]; keccak_gpu_state[11] ^= u[1]; keccak_gpu_state[16] ^= u[1]; keccak_gpu_state[21] ^= u[1];
			keccak_gpu_state[2] ^= u[2]; keccak_gpu_state[7] ^= u[2]; keccak_gpu_state[12] ^= u[2]; keccak_gpu_state[17] ^= u[2]; keccak_gpu_state[22] ^= u[2];
			keccak_gpu_state[3] ^= u[3]; keccak_gpu_state[8] ^= u[3]; keccak_gpu_state[13] ^= u[3]; keccak_gpu_state[18] ^= u[3]; keccak_gpu_state[23] ^= u[3];
			keccak_gpu_state[4] ^= u[4]; keccak_gpu_state[9] ^= u[4]; keccak_gpu_state[14] ^= u[4]; keccak_gpu_state[19] ^= u[4]; keccak_gpu_state[24] ^= u[4];
			v = keccak_gpu_state[1];
			keccak_gpu_state[1] = SPH_ROTL64(keccak_gpu_state[6], 44);
			keccak_gpu_state[6] = SPH_ROTL64(keccak_gpu_state[9], 20);
			keccak_gpu_state[9] = SPH_ROTL64(keccak_gpu_state[22], 61);
			keccak_gpu_state[22] = SPH_ROTL64(keccak_gpu_state[14], 39);
			keccak_gpu_state[14] = SPH_ROTL64(keccak_gpu_state[20], 18);
			keccak_gpu_state[20] = SPH_ROTL64(keccak_gpu_state[2], 62);
			keccak_gpu_state[2] = SPH_ROTL64(keccak_gpu_state[12], 43);
			keccak_gpu_state[12] = SPH_ROTL64(keccak_gpu_state[13], 25);
			keccak_gpu_state[13] = SPH_ROTL64(keccak_gpu_state[19], 8);
			keccak_gpu_state[19] = SPH_ROTL64(keccak_gpu_state[23], 56);
			keccak_gpu_state[23] = SPH_ROTL64(keccak_gpu_state[15], 41);
			keccak_gpu_state[15] = SPH_ROTL64(keccak_gpu_state[4], 27);
			keccak_gpu_state[4] = SPH_ROTL64(keccak_gpu_state[24], 14);
			keccak_gpu_state[24] = SPH_ROTL64(keccak_gpu_state[21], 2);
			keccak_gpu_state[21] = SPH_ROTL64(keccak_gpu_state[8], 55);
			keccak_gpu_state[8] = SPH_ROTL64(keccak_gpu_state[16], 45);
			keccak_gpu_state[16] = SPH_ROTL64(keccak_gpu_state[5], 36);
			keccak_gpu_state[5] = SPH_ROTL64(keccak_gpu_state[3], 28);
			keccak_gpu_state[3] = SPH_ROTL64(keccak_gpu_state[18], 21);
			keccak_gpu_state[18] = SPH_ROTL64(keccak_gpu_state[17], 15);
			keccak_gpu_state[17] = SPH_ROTL64(keccak_gpu_state[11], 10);
			keccak_gpu_state[11] = SPH_ROTL64(keccak_gpu_state[7], 6);
			keccak_gpu_state[7] = SPH_ROTL64(keccak_gpu_state[10], 3);
			keccak_gpu_state[10] = SPH_ROTL64(v, 1);
			v = keccak_gpu_state[0]; w = keccak_gpu_state[1]; keccak_gpu_state[0] ^= (~w) & keccak_gpu_state[2]; keccak_gpu_state[1] ^= (~keccak_gpu_state[2]) & keccak_gpu_state[3]; keccak_gpu_state[2] ^= (~keccak_gpu_state[3]) & keccak_gpu_state[4]; keccak_gpu_state[3] ^= (~keccak_gpu_state[4]) & v; keccak_gpu_state[4] ^= (~v) & w;
			v = keccak_gpu_state[5]; w = keccak_gpu_state[6]; keccak_gpu_state[5] ^= (~w) & keccak_gpu_state[7]; keccak_gpu_state[6] ^= (~keccak_gpu_state[7]) & keccak_gpu_state[8]; keccak_gpu_state[7] ^= (~keccak_gpu_state[8]) & keccak_gpu_state[9]; keccak_gpu_state[8] ^= (~keccak_gpu_state[9]) & v; keccak_gpu_state[9] ^= (~v) & w;
			v = keccak_gpu_state[10]; w = keccak_gpu_state[11]; keccak_gpu_state[10] ^= (~w) & keccak_gpu_state[12]; keccak_gpu_state[11] ^= (~keccak_gpu_state[12]) & keccak_gpu_state[13]; keccak_gpu_state[12] ^= (~keccak_gpu_state[13]) & keccak_gpu_state[14]; keccak_gpu_state[13] ^= (~keccak_gpu_state[14]) & v; keccak_gpu_state[14] ^= (~v) & w;
			v = keccak_gpu_state[15]; w = keccak_gpu_state[16]; keccak_gpu_state[15] ^= (~w) & keccak_gpu_state[17]; keccak_gpu_state[16] ^= (~keccak_gpu_state[17]) & keccak_gpu_state[18]; keccak_gpu_state[17] ^= (~keccak_gpu_state[18]) & keccak_gpu_state[19]; keccak_gpu_state[18] ^= (~keccak_gpu_state[19]) & v; keccak_gpu_state[19] ^= (~v) & w;
			v = keccak_gpu_state[20]; w = keccak_gpu_state[21]; keccak_gpu_state[20] ^= (~w) & keccak_gpu_state[22]; keccak_gpu_state[21] ^= (~keccak_gpu_state[22]) & keccak_gpu_state[23]; keccak_gpu_state[22] ^= (~keccak_gpu_state[23]) & keccak_gpu_state[24]; keccak_gpu_state[23] ^= (~keccak_gpu_state[24]) & v; keccak_gpu_state[24] ^= (~v) & w;
			keccak_gpu_state[0] ^= RC[i];
	}
	hashB = (uint*)keccak_gpu_state;
    //cubehash32
	x0 = 0xEA2BD4B4; x1 = 0xCCD6F29F; x2 = 0x63117E71;
	x3 = 0x35481EAE; x4 = 0x22512D5B; x5 = 0xE5D94E63;
	x6 = 0x7E624131; x7 = 0xF4CC12BE; x8 = 0xC2D0B696;
	x9 = 0x42AF2070; xa = 0xD0720C35; xb = 0x3361DA8C;
	xc = 0x28CCECA4; xd = 0x8EF8AD83; xe = 0x4680AC00;
	xf = 0x40E5FBAB;
	xg = 0xD89041C3; xh = 0x6107FBD5;
	xi = 0x6C859D41; xj = 0xF0B26679; xk = 0x09392549;
	xl = 0x5FA25603; xm = 0x65C892FD; xn = 0x93CB6285;
	xo = 0x2AF2B5AE; xp = 0x9E4B4E60; xq = 0x774ABFDD;
	xr = 0x85254725; xs = 0x15815AEB; xt = 0x4AB6AAD6;
	xu = 0x9CDAF8AF; xv = 0xD6032C0A;
	x0 ^= (hashB[0]);
	x1 ^= (hashB[1]);
	x2 ^= (hashB[2]);
	x3 ^= (hashB[3]);
	x4 ^= (hashB[4]);
	x5 ^= (hashB[5]);
	x6 ^= (hashB[6]);
	x7 ^= (hashB[7]);
	SIXTEEN_ROUNDS;
	x0 ^= 0x80;
	SIXTEEN_ROUNDS;
	xv ^= 0x01;
	for (int i = 0; i < 10; ++i) SIXTEEN_ROUNDS;
	hashedHeader[0] = x0;
	hashedHeader[1] = x1;
	hashedHeader[2] = x2;
	hashedHeader[3] = x3;
	hashedHeader[4] = x4;
	hashedHeader[5] = x5;
	hashedHeader[6] = x6;
	hashedHeader[7] = x7;
    //lyra2 start
	state = (ulong4*)hashedHeader;	
	state[1] = state[0]; state[2] = SPH_ULONG4(0x6a09e667f3bcc908UL, 0xbb67ae8584caa73bUL, 0x3c6ef372fe94f82bUL, 0xa54ff53a5f1d36f1UL); state[3] = SPH_ULONG4(0x510e527fade682d1UL, 0x9b05688c2b3e6c1fUL, 0x1f83d9abfb41bd6bUL, 0x5be0cd19137e2179UL);	for (int i = 0; i<12; i++) round_lyra(state); state[0] ^= SPH_ULONG4(0x20,0x20,0x20,0x01); state[1] ^= SPH_ULONG4(0x04,0x04,0x80,0x0100000000000000); for (int i = 0; i<12; i++) round_lyra(state); ps1 = (3 * 3); for (int i = 0; i < 4; i++) { uint s1 = ps1 - 3 * i; for (int j = 0; j < 3; j++) (DMatrix)[j+s1] = state[j]; round_lyra(state); }	
    // squashed reduceduplexf - reduceDuplexf(state,DMatrix)
	ps1 = 0; ps2 = (3 * 3 + 3 * 4); for (int i = 0; i < 4; i++) { uint s1 = ps1 + i*3; uint s2 = ps2 - i*3; for (int j = 0; j < 3; j++) state1[j] = (DMatrix)[j + s1]; for (int j = 0; j < 3; j++) state[j] ^= state1[j]; round_lyra(state); for (int j = 0; j < 3; j++) state1[j] ^= state[j]; for (int j = 0; j < 3; j++) (DMatrix)[j + s2] = state1[j]; }	
	// squashed reduceduplexrowsetupf (1,0,2..) - reduceDuplexRowSetupf(uint rowIn, uint rowInOut, uint rowOut, ulong4 *state,  __global ulong4* DMatrix)
	ps1 = (3 * 4 * 1); ps2 = (3 * 4 * 0); ps3 = (3 * 3 + 3 * 4 * 2); for (int i = 0; i < 4; i++) { uint s1 = ps1 + i*3; uint s2 = ps2 + i*3; uint s3 = ps3 - i*3;	for (int j = 0; j < 3; j++) state1[j] = (DMatrix)[j + s1]; for (int j = 0; j < 3; j++) state2[j] = (DMatrix)[j + s2]; for (int j = 0; j < 3; j++) { ulong4 tmp = state1[j] + state2[j]; state[j] ^= tmp; } round_lyra(state); for (int j = 0; j < 3; j++) { state1[j] ^= state[j]; (DMatrix)[j + s3] = state1[j]; } ((ulong*)state2)[0] ^= ((ulong*)state)[11];	for (int j = 0; j < 11; j++) ((ulong*)state2)[j + 1] ^= ((ulong*)state)[j];	for (int j = 0; j < 3; j++) (DMatrix)[j + s2] = state2[j]; }
	// squashed reduceduplexrowsetupf (2,1,3..) - reduceDuplexRowSetupf(uint rowIn, uint rowInOut, uint rowOut, ulong4 *state,  __global ulong4* DMatrix)
	ps1 = (3 * 4 * 2); ps2 = (3 * 4 * 1); ps3 = (3 * 3 + 3 * 4 * 3); for (int i = 0; i < 4; i++) { uint s1 = ps1 + i*3; uint s2 = ps2 + i*3; uint s3 = ps3 - i*3;	for (int j = 0; j < 3; j++) state1[j] = (DMatrix)[j + s1]; for (int j = 0; j < 3; j++) state2[j] = (DMatrix)[j + s2]; for (int j = 0; j < 3; j++) { ulong4 tmp = state1[j] + state2[j]; state[j] ^= tmp; } round_lyra(state); for (int j = 0; j < 3; j++) { state1[j] ^= state[j]; (DMatrix)[j + s3] = state1[j]; } ((ulong*)state2)[0] ^= ((ulong*)state)[11];	for (int j = 0; j < 11; j++) ((ulong*)state2)[j + 1] ^= ((ulong*)state)[j];	for (int j = 0; j < 3; j++) (DMatrix)[j + s2] = state2[j]; }
    // last loops of lyra2
	prev = 3; for (uint h = 0; h<4; h++) { rowa = state[0].x & 3; ulong4 state1[3], state2[3]; uint ps1 = (3 * 4 * prev);   uint ps2 = (3 * 4 * rowa); uint ps3 = (3 * 4 * h); for (int i = 0; i < 4; i++) { uint s1 = ps1 + i*3; uint s2 = ps2 + i*3; uint s3 = ps3 + i*3; for (int j = 0; j < 3; j++) state1[j] = (DMatrix)[j + s1]; for (int j = 0; j < 3; j++) state2[j] = (DMatrix)[j + s2]; for (int j = 0; j < 3; j++) state1[j] += state2[j]; for (int j = 0; j < 3; j++) state[j] ^= state1[j]; round_lyra(state); ((ulong*)state2)[0] ^= ((ulong*)state)[11]; for (int j = 0; j < 11; j++) ((ulong*)state2)[j + 1] ^= ((ulong*)state)[j];      if (rowa != h) { for (int j = 0; j < 3; j++) (DMatrix)[j + s2] = state2[j]; for (int j = 0; j < 3; j++) (DMatrix)[j + s3] ^= state[j];  } else { for (int j = 0; j < 3; j++) state2[j] ^= state[j]; for (int j = 0; j < 3; j++) (DMatrix)[j + s2] = state2[j]; }} prev = h; } shift = (3 * 4 * rowa);	for (int j = 0; j < 3; j++) state[j] ^= (DMatrix)[j+shift]; for (int i = 0; i < 12; i++) round_lyra(state);	
	hashB = (uint*)state;
    //keccak32
        in_dwords = (ulong*)hashB;
        h[8] = skein_ks_parity;
        for (int i = 0; i<8; i++) { h[i] = SKEIN_IV512_256[i]; h[8] ^= h[i]; }  t[0]=t12[0]; t[1]=t12[1]; t[2]=t12[2];
        dt0= (in_dwords[0]); dt1= (in_dwords[1]); dt2= (in_dwords[2]); dt3= (in_dwords[3]);
        p0 = h[0] + dt0;p1 = h[1] + dt1;p2 = h[2] + dt2;p3 = h[3] + dt3;p4 = h[4];p5 = h[5] + t[0];p6 = h[6] + t[1];p7 = h[7];
        for (int i = 1; i<19; i+=2) {Round_8_512(p0,p1,p2,p3,p4,p5,p6,p7,i);}
        p0 ^= dt0; p1 ^= dt1; p2 ^= dt2; p3 ^= dt3;
        h[0] = p0;h[1] = p1;h[2] = p2;h[3] = p3;h[4] = p4;h[5] = p5;h[6] = p6;h[7] = p7;h[8] = skein_ks_parity;
        for (int i = 0; i<8; i++) { h[8] ^= h[i]; } t[0] = t12[3];t[1] = t12[4];t[2] = t12[5];p5 += t[0];p6 += t[1];
        for (int i = 1; i<19; i+=2) { Round_8_512(p0, p1, p2, p3, p4, p5, p6, p7, i); }
        out_dwords[0] = (p0);out_dwords[1] = (p1);out_dwords[2] = (p2); out_dwords[3] = (p3);
        hashB = (uint*)out_dwords;
    //cubehash32
	x0 = 0xEA2BD4B4; x1 = 0xCCD6F29F; x2 = 0x63117E71;
	x3 = 0x35481EAE; x4 = 0x22512D5B; x5 = 0xE5D94E63;
	x6 = 0x7E624131; x7 = 0xF4CC12BE; x8 = 0xC2D0B696;
	x9 = 0x42AF2070; xa = 0xD0720C35; xb = 0x3361DA8C;
	xc = 0x28CCECA4; xd = 0x8EF8AD83; xe = 0x4680AC00;
	xf = 0x40E5FBAB;
	xg = 0xD89041C3; xh = 0x6107FBD5;
	xi = 0x6C859D41; xj = 0xF0B26679; xk = 0x09392549;
	xl = 0x5FA25603; xm = 0x65C892FD; xn = 0x93CB6285;
	xo = 0x2AF2B5AE; xp = 0x9E4B4E60; xq = 0x774ABFDD;
	xr = 0x85254725; xs = 0x15815AEB; xt = 0x4AB6AAD6;
	xu = 0x9CDAF8AF; xv = 0xD6032C0A;
	x0 ^= (hashB[0]);
	x1 ^= (hashB[1]);
	x2 ^= (hashB[2]);
	x3 ^= (hashB[3]);
	x4 ^= (hashB[4]);
	x5 ^= (hashB[5]);
	x6 ^= (hashB[6]);
	x7 ^= (hashB[7]);
	SIXTEEN_ROUNDS;
	x0 ^= 0x80;
	SIXTEEN_ROUNDS;
	xv ^= 0x01;
	for (int i = 0; i < 10; ++i) SIXTEEN_ROUNDS;
    //bmw32
dh[0] = 0x40414243; dh[1] = 0x44454647; dh[2] = 0x48494A4B; dh[3] = 0x4C4D4E4F;
dh[4] = 0x50515253; dh[5] = 0x54555657; dh[6] = 0x58595A5B; dh[7] = 0x5C5D5E5F;
dh[8] = 0x60616263; dh[9] = 0x64656667; dh[10] = 0x68696A6B; dh[11] = 0x6C6D6E6F;
dh[12] = 0x70717273; dh[13] = 0x74757677; dh[14] = 0x78797A7B; dh[15] = 0x7C7D7E7F;
final_s[0] = 0xaaaaaaa0; final_s[1] = 0xaaaaaaa1; final_s[2] = 0xaaaaaaa2; final_s[3] = 0xaaaaaaa3; 
final_s[4] = 0xaaaaaaa4; final_s[5] = 0xaaaaaaa5; final_s[6] = 0xaaaaaaa6; final_s[7] = 0xaaaaaaa7; 
final_s[8] = 0xaaaaaaa8; final_s[9] = 0xaaaaaaa9; final_s[10] = 0xaaaaaaaa; final_s[11] = 0xaaaaaaab;
final_s[12] = 0xaaaaaaac; final_s[13] = 0xaaaaaaad; final_s[14] = 0xaaaaaaae; final_s[15] = 0xaaaaaaaf;
        message[0] = x0; message[1] = x1; message[2] = x2; message[3] = x3; message[4] = x4;
        message[5] = x5; message[6] = x6; message[7] = x7; message[9] = 0; message[10] = 0;
        message[11] = 0; message[12] = 0; message[13] = 0; message[8]= 0x80; message[14]=0x100; message[15]=0;
        //compression256 message_dh_start
        Q[0]=ss0((message[5]^dh[5])-(message[7]^dh[7])+(message[10]^dh[10])+(message[13]^dh[13])+(message[14]^dh[14]))+dh[1];
        Q[1]=ss1((message[6]^dh[6])-(message[8]^dh[8])+(message[11]^dh[11])+(message[14]^dh[14])-(message[15]^dh[15]))+dh[2];
        Q[2]=ss2((message[0]^dh[0])+(message[7]^dh[7])+(message[9]^dh[9])-(message[12]^dh[12])+(message[15]^dh[15]))+dh[3];
        Q[3]=ss3((message[0]^dh[0])-(message[1]^dh[1])+(message[8]^dh[8])-(message[10]^dh[10])+(message[13]^dh[13]))+dh[4];
        Q[4]=ss4((message[1]^dh[1])+(message[2]^dh[2])+(message[9]^dh[9])-(message[11]^dh[11])-(message[14]^dh[14]))+dh[5];
        Q[5]=ss0((message[3]^dh[3])-(message[2]^dh[2])+(message[10]^dh[10])-(message[12]^dh[12])+(message[15]^dh[15]))+dh[6];
        Q[6]=ss1((message[4]^dh[4])-(message[0]^dh[0])-(message[3]^dh[3])-(message[11]^dh[11])+(message[13]^dh[13]))+dh[7];
        Q[7]=ss2((message[1]^dh[1])-(message[4]^dh[4])-(message[5]^dh[5])-(message[12]^dh[12])-(message[14]^dh[14]))+dh[8];
        Q[8]=ss3((message[2]^dh[2])-(message[5]^dh[5])-(message[6]^dh[6])+(message[13]^dh[13])-(message[15]^dh[15]))+dh[9];
        Q[9]=ss4((message[0]^dh[0])-(message[3]^dh[3])+(message[6]^dh[6])-(message[7]^dh[7])+(message[14]^dh[14]))+dh[10];
        Q[10]=ss0((message[8]^dh[8])-(message[1]^dh[1])-(message[4]^dh[4])-(message[7]^dh[7])+(message[15]^dh[15]))+dh[11];
        Q[11]=ss1((message[8]^dh[8])-(message[0]^dh[0])-(message[2]^dh[2])-(message[5]^dh[5])+(message[9]^dh[9]))+dh[12];
        Q[12]=ss2((message[1]^dh[1])+(message[3]^dh[3])-(message[6]^dh[6])-(message[9]^dh[9])+(message[10]^dh[10]))+dh[13];
        Q[13]=ss3((message[2]^dh[2])+(message[4]^dh[4])+(message[7]^dh[7])+(message[10]^dh[10])+(message[11]^dh[11]))+dh[14];
        Q[14]=ss4((message[3]^dh[3])-(message[5]^dh[5])+(message[8]^dh[8])-(message[11]^dh[11])-(message[12]^dh[12]))+dh[15];
        Q[15]=ss0((message[12]^dh[12])-(message[4]^dh[4])-(message[6]^dh[6])-(message[9]^dh[9])+(message[13]^dh[13]))+dh[0];
        Q[16]=expand32_1(16,message,dh,Q);
        Q[17]=expand32_1(17,message,dh,Q);
        Q[18]=expand32_2(18,message,dh,Q);
        Q[19]=expand32_2(19,message,dh,Q);
        Q[20]=expand32_2(20,message,dh,Q);
        Q[21]=expand32_2(21,message,dh,Q);
        Q[22]=expand32_2(22,message,dh,Q);
        Q[23]=expand32_2(23,message,dh,Q);
        Q[24]=expand32_2(24,message,dh,Q);
        Q[25]=expand32_2(25,message,dh,Q);
        Q[26]=expand32_2(26,message,dh,Q);
        Q[27]=expand32_2(27,message,dh,Q);
        Q[28]=expand32_2(28,message,dh,Q);
        Q[29]=expand32_2(29,message,dh,Q);
        Q[30]=expand32_2(30,message,dh,Q);
        Q[31]=expand32_2(31,message,dh,Q);
        XL32=Q[16]^Q[17]^Q[18]^Q[19]^Q[20]^Q[21]^Q[22]^Q[23];
        XH32=XL32^Q[24]^Q[25]^Q[26]^Q[27]^Q[28]^Q[29]^Q[30]^Q[31];
        dh[0]=(shl(XH32,5)^shr(Q[16],5)^message[0])+(XL32^Q[24]^Q[0]);
        dh[1]=(shr(XH32,7)^shl(Q[17],8)^message[1])+(XL32^Q[25]^Q[1]);
        dh[2]=(shr(XH32,5)^shl(Q[18],5)^message[2])+(XL32^Q[26]^Q[2]);
        dh[3]=(shr(XH32,1)^shl(Q[19],5)^message[3])+(XL32^Q[27]^Q[3]);
        dh[4]=(shr(XH32,3)^Q[20]^message[4])+(XL32^Q[28]^Q[4]);
        dh[5]=(shl(XH32,6)^shr(Q[21],6)^message[5])+(XL32^Q[29]^Q[5]);
        dh[6]=(shr(XH32,4)^shl(Q[22],6)^message[6])+(XL32^Q[30]^Q[6]);
        dh[7]=(shr(XH32,11)^shl(Q[23],2)^message[7])+(XL32^Q[31]^Q[7]);
        dh[8]=SPH_ROTL32(dh[4],9)+(XH32^Q[24]^message[8])+(shl(XL32,8)^Q[23]^Q[8]);
        dh[9]=SPH_ROTL32(dh[5],10)+(XH32^Q[25]^message[9])+(shr(XL32,6)^Q[16]^Q[9]);
        dh[10]=SPH_ROTL32(dh[6],11)+(XH32^Q[26]^message[10])+(shl(XL32,6)^Q[17]^Q[10]);
        dh[11]=SPH_ROTL32(dh[7],12)+(XH32^Q[27]^message[11])+(shl(XL32,4)^Q[18]^Q[11]);
        dh[12]=SPH_ROTL32(dh[0],13)+(XH32^Q[28]^message[12])+(shr(XL32,3)^Q[19]^Q[12]);
        dh[13]=SPH_ROTL32(dh[1],14)+(XH32^Q[29]^message[13])+(shr(XL32,4)^Q[20]^Q[13]);
        dh[14]=SPH_ROTL32(dh[2],15)+(XH32^Q[30]^message[14])+(shr(XL32,7)^Q[21]^Q[14]);
        dh[15]=SPH_ROTL32(dh[3],16)+(XH32^Q[31]^message[15])+(shr(XL32,2)^Q[22]^Q[15]);
        //compression256 message_dh_end
	//compression256 dh_finals_start
	Q[0]=ss0((dh[5]^final_s[5])-(dh[7]^final_s[7])+(dh[10]^final_s[10])+(dh[13]^final_s[13])+(dh[14]^final_s[14]))+final_s[1];
	Q[1]=ss1((dh[6]^final_s[6])-(dh[8]^final_s[8])+(dh[11]^final_s[11])+(dh[14]^final_s[14])-(dh[15]^final_s[15]))+final_s[2];
	Q[2]=ss2((dh[0]^final_s[0])+(dh[7]^final_s[7])+(dh[9]^final_s[9])-(dh[12]^final_s[12])+(dh[15]^final_s[15]))+final_s[3];
	Q[3]=ss3((dh[0]^final_s[0])-(dh[1]^final_s[1])+(dh[8]^final_s[8])-(dh[10]^final_s[10])+(dh[13]^final_s[13]))+final_s[4];
	Q[4]=ss4((dh[1]^final_s[1])+(dh[2]^final_s[2])+(dh[9]^final_s[9])-(dh[11]^final_s[11])-(dh[14]^final_s[14]))+final_s[5];
	Q[5]=ss0((dh[3]^final_s[3])-(dh[2]^final_s[2])+(dh[10]^final_s[10])-(dh[12]^final_s[12])+(dh[15]^final_s[15]))+final_s[6];
	Q[6]=ss1((dh[4]^final_s[4])-(dh[0]^final_s[0])-(dh[3]^final_s[3])-(dh[11]^final_s[11])+(dh[13]^final_s[13]))+final_s[7];
	Q[7]=ss2((dh[1]^final_s[1])-(dh[4]^final_s[4])-(dh[5]^final_s[5])-(dh[12]^final_s[12])-(dh[14]^final_s[14]))+final_s[8];
	Q[8]=ss3((dh[2]^final_s[2])-(dh[5]^final_s[5])-(dh[6]^final_s[6])+(dh[13]^final_s[13])-(dh[15]^final_s[15]))+final_s[9];
	Q[9]=ss4((dh[0]^final_s[0])-(dh[3]^final_s[3])+(dh[6]^final_s[6])-(dh[7]^final_s[7])+(dh[14]^final_s[14]))+final_s[10];
	Q[10]=ss0((dh[8]^final_s[8])-(dh[1]^final_s[1])-(dh[4]^final_s[4])-(dh[7]^final_s[7])+(dh[15]^final_s[15]))+final_s[11];
	Q[11]=ss1((dh[8]^final_s[8])-(dh[0]^final_s[0])-(dh[2]^final_s[2])-(dh[5]^final_s[5])+(dh[9]^final_s[9]))+final_s[12];
	Q[12]=ss2((dh[1]^final_s[1])+(dh[3]^final_s[3])-(dh[6]^final_s[6])-(dh[9]^final_s[9])+(dh[10]^final_s[10]))+final_s[13];
	Q[13]=ss3((dh[2]^final_s[2])+(dh[4]^final_s[4])+(dh[7]^final_s[7])+(dh[10]^final_s[10])+(dh[11]^final_s[11]))+final_s[14];
	Q[14]=ss4((dh[3]^final_s[3])-(dh[5]^final_s[5])+(dh[8]^final_s[8])-(dh[11]^final_s[11])-(dh[12]^final_s[12]))+final_s[15];
	Q[15]=ss0((dh[12]^final_s[12])-(dh[4]^final_s[4])-(dh[6]^final_s[6])-(dh[9]^final_s[9])+(dh[13]^final_s[13]))+final_s[0];
	Q[16]=expand32_1(16,dh,final_s,Q);
	Q[17]=expand32_1(17,dh,final_s,Q);
	Q[18]=expand32_2(18,dh,final_s,Q);
	Q[19]=expand32_2(19,dh,final_s,Q);
	Q[20]=expand32_2(20,dh,final_s,Q);
	Q[21]=expand32_2(21,dh,final_s,Q);
	Q[22]=expand32_2(22,dh,final_s,Q);
	Q[23]=expand32_2(23,dh,final_s,Q);
	Q[24]=expand32_2(24,dh,final_s,Q);
	Q[25]=expand32_2(25,dh,final_s,Q);
	Q[26]=expand32_2(26,dh,final_s,Q);
	Q[27]=expand32_2(27,dh,final_s,Q);
	Q[28]=expand32_2(28,dh,final_s,Q);
	Q[29]=expand32_2(29,dh,final_s,Q);
	Q[30]=expand32_2(30,dh,final_s,Q);
	Q[31]=expand32_2(31,dh,final_s,Q);
	XL32=Q[16]^Q[17]^Q[18]^Q[19]^Q[20]^Q[21]^Q[22]^Q[23];
	XH32=XL32^Q[24]^Q[25]^Q[26]^Q[27]^Q[28]^Q[29]^Q[30]^Q[31];
	final_s[0]=(shl(XH32,5)^shr(Q[16],5)^dh[0])+(XL32^Q[24]^Q[0]);
	final_s[1]=(shr(XH32,7)^shl(Q[17],8)^dh[1])+(XL32^Q[25]^Q[1]);
	final_s[2]=(shr(XH32,5)^shl(Q[18],5)^dh[2])+(XL32^Q[26]^Q[2]);
	final_s[3]=(shr(XH32,1)^shl(Q[19],5)^dh[3])+(XL32^Q[27]^Q[3]);
	final_s[4]=(shr(XH32,3)^Q[20]^dh[4])+(XL32^Q[28]^Q[4]);
	final_s[5]=(shl(XH32,6)^shr(Q[21],6)^dh[5])+(XL32^Q[29]^Q[5]);
	final_s[6]=(shr(XH32,4)^shl(Q[22],6)^dh[6])+(XL32^Q[30]^Q[6]);
	final_s[7]=(shr(XH32,11)^shl(Q[23],2)^dh[7])+(XL32^Q[31]^Q[7]);
	final_s[8]=SPH_ROTL32(final_s[4],9)+(XH32^Q[24]^dh[8])+(shl(XL32,8)^Q[23]^Q[8]);
	final_s[9]=SPH_ROTL32(final_s[5],10)+(XH32^Q[25]^dh[9])+(shr(XL32,6)^Q[16]^Q[9]);
	final_s[10]=SPH_ROTL32(final_s[6],11)+(XH32^Q[26]^dh[10])+(shl(XL32,6)^Q[17]^Q[10]);
	final_s[11]=SPH_ROTL32(final_s[7],12)+(XH32^Q[27]^dh[11])+(shl(XL32,4)^Q[18]^Q[11]);
	final_s[12]=SPH_ROTL32(final_s[0],13)+(XH32^Q[28]^dh[12])+(shr(XL32,3)^Q[19]^Q[12]);
	final_s[13]=SPH_ROTL32(final_s[1],14)+(XH32^Q[29]^dh[13])+(shr(XL32,4)^Q[20]^Q[13]);
	final_s[14]=SPH_ROTL32(final_s[2],15)+(XH32^Q[30]^dh[14])+(shr(XL32,7)^Q[21]^Q[14]);
	final_s[15]=SPH_ROTL32(final_s[3],16)+(XH32^Q[31]^dh[15])+(shr(XL32,2)^Q[22]^Q[15]);
	//compression256 dh_finals_end

	ulong check = as_ulong((uint2)(final_s[14], final_s[15]));

	// target itself should be in little-endian format, 
#ifdef NVIDIA

	if (check <= target){
                block[0] = (final_s[8]);
                block[1] = (final_s[9]);
                block[2] = (final_s[10]);
                block[3] = (final_s[11]);
                block[4] = (final_s[12]);
                block[5] = (final_s[13]);
                block[6] = (final_s[14]);
                block[7] = (final_s[15]);
		uint slot = atomic_inc(&g_output[MAX_OUTPUTS]);
		g_output[slot & MAX_OUTPUTS] = gid;
	}
#else
	if (check <= target){
		uint slot = min(MAX_OUTPUTS-1u, convert_uint(atomic_inc(&g_output[MAX_OUTPUTS])));
		g_output[slot] = gid;
	}
#endif
}

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
