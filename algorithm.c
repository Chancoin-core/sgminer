/*
 * Copyright 2014 sgminer developers
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or (at
 * your option) any later version.  See COPYING for more details.
 */

#include "algorithm.h"
#include "sph/sph_sha2.h"
#include "ocl.h"
#include "ocl/build_kernel.h"

#include "algorithm/scrypt.h"
#include "algorithm/animecoin.h"
#include "algorithm/inkcoin.h"
#include "algorithm/quarkcoin.h"
#include "algorithm/qubitcoin.h"
#include "algorithm/sifcoin.h"
#include "algorithm/darkcoin.h"
#include "algorithm/myriadcoin-groestl.h"
#include "algorithm/fuguecoin.h"
#include "algorithm/groestlcoin.h"
#include "algorithm/twecoin.h"
#include "algorithm/marucoin.h"
#include "algorithm/maxcoin.h"
#include "algorithm/talkcoin.h"
#include "algorithm/bitblock.h"
#include "algorithm/x14.h"
#include "algorithm/fresh.h"
#include "algorithm/whirlcoin.h"
#include "algorithm/neoscrypt.h"
#include "algorithm/whirlpoolx.h"
#include "algorithm/lyra2re.h"
#include "algorithm/lyra2rev2.h"
#include "algorithm/pluck.h"
#include "algorithm/yescrypt.h"
#include "algorithm/credits.h"
#include "algorithm/blake256.h"
#include "algorithm/blakecoin.h"
#include "algorithm/decred.h"
#include "algorithm/ethash.h"
#include "algorithm/sibcoin.h"
#include "algorithm/skein2.h"
#include "algorithm/skunk.h"
#include "algorithm/tribus.h"
#include "algorithm/veltor.h"
#include "algorithm/nightcap.h"


#include "compat.h"

#include <inttypes.h>
#include <string.h>
#include <assert.h>

const char *algorithm_type_str[] = {
  "Unknown",
  "Credits",
  "Scrypt",
  "NScrypt",
  "X11",
  "X13",
  "X14",
  "X15",
  "Keccak",
  "Quarkcoin",
  "Twecoin",
  "Fugue256",
  "NIST",
  "Fresh",
  "Whirlcoin",
  "Neoscrypt",
  "WhirlpoolX",
  "Lyra2RE",
  "Lyra2REV2"
  "Pluck"
  "Yescrypt",
  "Yescrypt-multi",
  "Blakecoin",
  "Blake",
  "Decred",
  "Vanilla",
  "Sibcoin",
  "Skein2",
  "Skunk",
  "Tribus",
  "Veltor",
  "Ethash",
  NULL
};

void sha256(const unsigned char *message, unsigned int len, unsigned char *digest)
{
  sph_sha256_context ctx_sha2;

  sph_sha256_init(&ctx_sha2);
  sph_sha256(&ctx_sha2, message, len);
  sph_sha256_close(&ctx_sha2, (void*)digest);
}

void gen_hash(const unsigned char *data, unsigned int len, unsigned char *hash)
{
  unsigned char hash1[32];
  sph_sha256_context ctx_sha2;

  sph_sha256_init(&ctx_sha2);
  sph_sha256(&ctx_sha2, data, len);
  sph_sha256_close(&ctx_sha2, hash1);
  sph_sha256(&ctx_sha2, hash1, 32);
  sph_sha256_close(&ctx_sha2, hash);
}

void raw_merkle(const unsigned char *coinbase, unsigned int len, unsigned char *merkle)
{
  memcpy(merkle, coinbase, len);
}

#define CL_SET_BLKARG(blkvar) status |= clSetKernelArg(*kernel, num++, sizeof(uint), (void *)&blk->blkvar)
#define CL_SET_VARG(args, var) status |= clSetKernelArg(*kernel, num++, args * sizeof(uint), (void *)var)
#define CL_SET_ARG_N(n, var) do { status |= clSetKernelArg(*kernel, n, sizeof(var), (void *)&var); } while (0)
#define CL_SET_ARG_0(var) CL_SET_ARG_N(0, var)
#define CL_SET_ARG(var) CL_SET_ARG_N(num++, var)
#define CL_NEXTKERNEL_SET_ARG_N(n, var) do { kernel++; CL_SET_ARG_N(n, var); } while (0)
#define CL_NEXTKERNEL_SET_ARG_0(var) CL_NEXTKERNEL_SET_ARG_N(0, var)
#define CL_NEXTKERNEL_SET_ARG(var) CL_NEXTKERNEL_SET_ARG_N(num++, var)

static void append_scrypt_compiler_options(struct _build_kernel_data *data, struct cgpu_info *cgpu, struct _algorithm_t *algorithm)
{
  char buf[255];
  sprintf(buf, " -D LOOKUP_GAP=%d -D CONCURRENT_THREADS=%u -D NFACTOR=%d",
    cgpu->lookup_gap, (unsigned int)cgpu->thread_concurrency, algorithm->nfactor);
  strcat(data->compiler_options, buf);

  sprintf(buf, "lg%utc%unf%u", cgpu->lookup_gap, (unsigned int)cgpu->thread_concurrency, algorithm->nfactor);
  strcat(data->binary_filename, buf);
}

static void append_ethash_compiler_options(struct _build_kernel_data *data, struct cgpu_info *cgpu, struct _algorithm_t *algorithm)
{
  char buf[255] = { 0 };
  bool nvidia = cgpu->has_nvml;
  if (!nvidia && cgpu->name)
    nvidia = strstr(cgpu->name, "GTX") || strstr(cgpu->name, "GeForce");
  if (nvidia)
    sprintf(buf, " -DNVIDIA ");
  strcat(data->compiler_options, buf);
  applog(LOG_DEBUG, "ETH compiler options for %s : %s", cgpu->name, data->compiler_options);
}

static void append_nightcap_compiler_options(struct _build_kernel_data *data, struct cgpu_info *cgpu, struct _algorithm_t *algorithm)
{
  char buf[255] = { 0 };
  bool nvidia = cgpu->has_nvml;
  if (!nvidia && cgpu->name)
    nvidia = strstr(cgpu->name, "GTX") || strstr(cgpu->name, "GeForce");
  if (nvidia)
    sprintf(buf, " -DNVIDIA ");
  strcat(data->compiler_options, buf);
  applog(LOG_DEBUG, "NIGHTCAP compiler options for %s : %s", cgpu->name, data->compiler_options);
}

static void append_neoscrypt_compiler_options(struct _build_kernel_data *data, struct cgpu_info *cgpu, struct _algorithm_t *algorithm)
{
  char buf[255];
  sprintf(buf, " %s-D MAX_GLOBAL_THREADS=%lu ",
    ((cgpu->lookup_gap > 0) ? " -D LOOKUP_GAP=2 " : ""), (unsigned long)cgpu->thread_concurrency);
  strcat(data->compiler_options, buf);

  sprintf(buf, "%stc%lu", ((cgpu->lookup_gap > 0) ? "lg" : ""), (unsigned long)cgpu->thread_concurrency);
  strcat(data->binary_filename, buf);
}

static void append_blake256_compiler_options(struct _build_kernel_data *data, struct cgpu_info *cgpu, struct _algorithm_t *algorithm)
{
  char buf[255];
  sprintf(buf, " -D LOOKUP_GAP=%d -D MAX_GLOBAL_THREADS=%lu ",
    cgpu->lookup_gap, (unsigned long)cgpu->thread_concurrency);
  strcat(data->compiler_options, buf);

  sprintf(buf, "tc%lu", (unsigned long)cgpu->thread_concurrency);
  strcat(data->binary_filename, buf);
}

static void append_x11_compiler_options(struct _build_kernel_data *data, struct cgpu_info *cgpu, struct _algorithm_t *algorithm)
{
  char buf[255];
  sprintf(buf, " -D SPH_COMPACT_BLAKE_64=%d -D SPH_LUFFA_PARALLEL=%d -D SPH_KECCAK_UNROLL=%u ",
    ((opt_blake_compact) ? 1 : 0), ((opt_luffa_parallel) ? 1 : 0), (unsigned int)opt_keccak_unroll);
  strcat(data->compiler_options, buf);

  sprintf(buf, "ku%u%s%s", (unsigned int)opt_keccak_unroll, ((opt_blake_compact) ? "bc" : ""), ((opt_luffa_parallel) ? "lp" : ""));
  strcat(data->binary_filename, buf);
}


static void append_x13_compiler_options(struct _build_kernel_data *data, struct cgpu_info *cgpu, struct _algorithm_t *algorithm)
{
  char buf[255];

  append_x11_compiler_options(data, cgpu, algorithm);

  sprintf(buf, " -D SPH_HAMSI_EXPAND_BIG=%d -D SPH_HAMSI_SHORT=%d ",
    (unsigned int)opt_hamsi_expand_big, ((opt_hamsi_short) ? 1 : 0));
  strcat(data->compiler_options, buf);

  sprintf(buf, "big%u%s", (unsigned int)opt_hamsi_expand_big, ((opt_hamsi_short) ? "hs" : ""));
  strcat(data->binary_filename, buf);
}

static cl_int queue_scrypt_kernel(struct __clState *clState, struct _dev_blk_ctx *blk, __maybe_unused cl_uint threads)
{
  unsigned char *midstate = blk->work->midstate;
  cl_kernel *kernel = &clState->kernel;
  unsigned int num = 0;
  cl_uint le_target;
  cl_int status = 0;

  le_target = *(cl_uint *)(blk->work->device_target + 28);
  memcpy(clState->cldata, blk->work->data, 80);
  status = clEnqueueWriteBuffer(clState->commandQueue, clState->CLbuffer0, true, 0, 80, clState->cldata, 0, NULL, NULL);

  CL_SET_ARG(clState->CLbuffer0);
  CL_SET_ARG(clState->outputBuffer);
  CL_SET_ARG(clState->padbuffer8);
  CL_SET_VARG(4, &midstate[0]);
  CL_SET_VARG(4, &midstate[16]);
  CL_SET_ARG(le_target);

  return status;
}

static cl_int queue_neoscrypt_kernel(_clState *clState, dev_blk_ctx *blk, __maybe_unused cl_uint threads)
{
  cl_kernel *kernel = &clState->kernel;
  unsigned int num = 0;
  cl_uint le_target;
  cl_int status = 0;

  /* This looks like a unnecessary double cast, but to make sure, that
   * the target's most significant entry is adressed as a 32-bit value
   * and not accidently by something else the double cast seems wise.
   * The compiler will get rid of it anyway. */
  le_target = (cl_uint)le32toh(((uint32_t *)blk->work->/*device_*/target)[7]);
  memcpy(clState->cldata, blk->work->data, 80);
  status = clEnqueueWriteBuffer(clState->commandQueue, clState->CLbuffer0, true, 0, 80, clState->cldata, 0, NULL, NULL);

  CL_SET_ARG(clState->CLbuffer0);
  CL_SET_ARG(clState->outputBuffer);
  CL_SET_ARG(clState->padbuffer8);
  CL_SET_ARG(le_target);

  return status;
}

static cl_int queue_credits_kernel(_clState *clState, dev_blk_ctx *blk, __maybe_unused cl_uint threads)
{
  cl_kernel *kernel = &clState->kernel;
  unsigned int num = 0;
  cl_ulong le_target;
  cl_int status = 0;


    // le_target = (*(cl_uint *)(blk->work->device_target + 24));
  le_target = (cl_ulong)le64toh(((uint64_t *)blk->work->/*device_*/target)[3]);
  //  le_target = (cl_uint)((uint32_t *)blk->work->target)[6];


  memcpy(clState->cldata, blk->work->data, 168);
//  flip168(clState->cldata, blk->work->data);
  status = clEnqueueWriteBuffer(clState->commandQueue, clState->CLbuffer0, true, 0, 168, clState->cldata, 0, NULL, NULL);

  CL_SET_ARG(clState->CLbuffer0);
  CL_SET_ARG(clState->outputBuffer);
  CL_SET_ARG(le_target);
  CL_SET_ARG(blk->work->midstate);

  return status;
}

static cl_int queue_yescrypt_kernel(_clState *clState, dev_blk_ctx *blk, __maybe_unused cl_uint threads)
{
  cl_kernel *kernel = &clState->kernel;
  unsigned int num = 0;
  cl_uint le_target;
  cl_int status = 0;


//  le_target = (*(cl_uint *)(blk->work->device_target + 28));
  le_target = (cl_uint)le32toh(((uint32_t *)blk->work->/*device_*/target)[7]);
//  le_target = (cl_uint)((uint32_t *)blk->work->target)[7];


//  memcpy(clState->cldata, blk->work->data, 80);
  flip80(clState->cldata, blk->work->data);
  status = clEnqueueWriteBuffer(clState->commandQueue, clState->CLbuffer0, true, 0, 80, clState->cldata, 0, NULL, NULL);

  CL_SET_ARG(clState->CLbuffer0);
  CL_SET_ARG(clState->outputBuffer);
  CL_SET_ARG(clState->padbuffer8);
  CL_SET_ARG(clState->buffer1);
  CL_SET_ARG(clState->buffer2);
  CL_SET_ARG(le_target);

  return status;
}

static cl_int queue_yescrypt_multikernel(_clState *clState, dev_blk_ctx *blk, __maybe_unused cl_uint threads)
{
//  cl_kernel *kernel = &clState->kernel;
  cl_kernel *kernel;
  unsigned int num = 0;
  cl_uint le_target;
  cl_int status = 0;


  //  le_target = (*(cl_uint *)(blk->work->device_target + 28));
  le_target = (cl_uint)le32toh(((uint32_t *)blk->work->/*device_*/target)[7]);
  memcpy(clState->cldata, blk->work->data, 80);
//  flip80(clState->cldata, blk->work->data);
  status = clEnqueueWriteBuffer(clState->commandQueue, clState->CLbuffer0, true, 0, 80, clState->cldata, 0, NULL, NULL);
//pbkdf and initial sha
  kernel = &clState->kernel;

  CL_SET_ARG(clState->CLbuffer0);
  CL_SET_ARG(clState->outputBuffer);
  CL_SET_ARG(clState->padbuffer8);
  CL_SET_ARG(clState->buffer1);
  CL_SET_ARG(clState->buffer2);
  CL_SET_ARG(clState->buffer3);
  CL_SET_ARG(le_target);

//inactive kernel
  num = 0;
  kernel = clState->extra_kernels;
  CL_SET_ARG_N(0,clState->buffer1);
  CL_SET_ARG_N(1,clState->buffer2);
//  CL_SET_ARG_N(3, clState->buffer3);

//mix2_2
  num = 0;
  CL_NEXTKERNEL_SET_ARG_N(0, clState->padbuffer8);
  CL_SET_ARG_N(1,clState->buffer1);
  CL_SET_ARG_N(2,clState->buffer2);
  //mix2_2
//inactive kernel
  num = 0;
  CL_NEXTKERNEL_SET_ARG_N(0, clState->buffer1);
  CL_SET_ARG_N(1, clState->buffer2);
  //mix2_2

  num = 0;
  CL_NEXTKERNEL_SET_ARG_N(0, clState->padbuffer8);
  CL_SET_ARG_N(1, clState->buffer1);
  CL_SET_ARG_N(2, clState->buffer2);

  //inactive kernel
  num = 0;
  CL_NEXTKERNEL_SET_ARG_N(0, clState->buffer1);
  CL_SET_ARG_N(1, clState->buffer2);
  //mix2_2


//pbkdf and finalization
    num=0;
  CL_NEXTKERNEL_SET_ARG(clState->CLbuffer0);
  CL_SET_ARG(clState->outputBuffer);
  CL_SET_ARG(clState->buffer2);
  CL_SET_ARG(clState->buffer3);
  CL_SET_ARG(le_target);

  return status;
}

static cl_int queue_maxcoin_kernel(struct __clState *clState, struct _dev_blk_ctx *blk, __maybe_unused cl_uint threads)
{
  cl_kernel *kernel = &clState->kernel;
  unsigned int num = 0;
  cl_int status = 0;

  flip80(clState->cldata, blk->work->data);
  status = clEnqueueWriteBuffer(clState->commandQueue, clState->CLbuffer0, true, 0, 80, clState->cldata, 0, NULL, NULL);

  CL_SET_ARG(clState->CLbuffer0);
  CL_SET_ARG(clState->outputBuffer);

  return status;
}

static cl_int queue_sph_kernel(struct __clState *clState, struct _dev_blk_ctx *blk, __maybe_unused cl_uint threads)
{
  cl_kernel *kernel = &clState->kernel;
  unsigned int num = 0;
  cl_ulong le_target;
  cl_int status = 0;

  le_target = *(cl_ulong *)(blk->work->device_target + 24);
  flip80(clState->cldata, blk->work->data);
  status = clEnqueueWriteBuffer(clState->commandQueue, clState->CLbuffer0, true, 0, 80, clState->cldata, 0, NULL, NULL);

  CL_SET_ARG(clState->CLbuffer0);
  CL_SET_ARG(clState->outputBuffer);
  CL_SET_ARG(le_target);

  return status;
}

static cl_int queue_darkcoin_mod_kernel(struct __clState *clState, struct _dev_blk_ctx *blk, __maybe_unused cl_uint threads)
{
  cl_kernel *kernel;
  unsigned int num;
  cl_ulong le_target;
  cl_int status = 0;

  le_target = *(cl_ulong *)(blk->work->device_target + 24);
  flip80(clState->cldata, blk->work->data);
  status = clEnqueueWriteBuffer(clState->commandQueue, clState->CLbuffer0, true, 0, 80, clState->cldata, 0, NULL, NULL);

  // blake - search
  kernel = &clState->kernel;
  num = 0;
  CL_SET_ARG(clState->CLbuffer0);
  CL_SET_ARG(clState->padbuffer8);
  // bmw - search1
  kernel = clState->extra_kernels;
  CL_SET_ARG_0(clState->padbuffer8);
  // groestl - search2
  CL_NEXTKERNEL_SET_ARG_0(clState->padbuffer8);
  // skein - search3
  CL_NEXTKERNEL_SET_ARG_0(clState->padbuffer8);
  // jh - search4
  CL_NEXTKERNEL_SET_ARG_0(clState->padbuffer8);
  // keccak - search5
  CL_NEXTKERNEL_SET_ARG_0(clState->padbuffer8);
  // luffa - search6
  CL_NEXTKERNEL_SET_ARG_0(clState->padbuffer8);
  // cubehash - search7
  CL_NEXTKERNEL_SET_ARG_0(clState->padbuffer8);
  // shavite - search8
  CL_NEXTKERNEL_SET_ARG_0(clState->padbuffer8);
  // simd - search9
  CL_NEXTKERNEL_SET_ARG_0(clState->padbuffer8);
  // echo - search10
  num = 0;
  CL_NEXTKERNEL_SET_ARG(clState->padbuffer8);
  CL_SET_ARG(clState->outputBuffer);
  CL_SET_ARG(le_target);

  return status;
}

static cl_int queue_sibcoin_kernel(struct __clState *clState, struct _dev_blk_ctx *blk, __maybe_unused cl_uint threads)
{
  cl_kernel *kernel;
  unsigned int num;
  cl_ulong le_target;
  cl_int status = 0;

  le_target = *(cl_ulong *)(blk->work->device_target + 24);
  flip80(clState->cldata, blk->work->data);
  status = clEnqueueWriteBuffer(clState->commandQueue, clState->CLbuffer0, true, 0, 80, clState->cldata, 0, NULL, NULL);

  // blake - search
  kernel = &clState->kernel;
  num = 0;
  CL_SET_ARG(clState->CLbuffer0);
  CL_SET_ARG(clState->padbuffer8);
  // bmw - search1
  kernel = clState->extra_kernels;
  CL_SET_ARG_0(clState->padbuffer8);
  // groestl - search2
  CL_NEXTKERNEL_SET_ARG_0(clState->padbuffer8);
  // skein - search3
  CL_NEXTKERNEL_SET_ARG_0(clState->padbuffer8);
  // jh - search4
  CL_NEXTKERNEL_SET_ARG_0(clState->padbuffer8);
  // keccak - search5
  CL_NEXTKERNEL_SET_ARG_0(clState->padbuffer8);
  // gost - search6
  CL_NEXTKERNEL_SET_ARG_0(clState->padbuffer8);
  // luffa - search7
  CL_NEXTKERNEL_SET_ARG_0(clState->padbuffer8);
  // cubehash - search8
  CL_NEXTKERNEL_SET_ARG_0(clState->padbuffer8);
  // shavite - search9
  CL_NEXTKERNEL_SET_ARG_0(clState->padbuffer8);
  // simd - search10
  CL_NEXTKERNEL_SET_ARG_0(clState->padbuffer8);
  // echo - search11
  num = 0;
  CL_NEXTKERNEL_SET_ARG(clState->padbuffer8);
  CL_SET_ARG(clState->outputBuffer);
  CL_SET_ARG(le_target);

  return status;
}

static cl_int queue_tribus_kernel(struct __clState *clState, struct _dev_blk_ctx *blk, __maybe_unused cl_uint threads)
{
  uint32_t *midstate = (uint32_t*)(&blk->work->midstate); // 128 bytes
  uint32_t *data_end = (uint32_t*)(&blk->work->data[64]); //  16 bytes (end of data)
  cl_kernel *kernel;
  cl_ulong le_target;
  cl_int status = 0;
  unsigned int num;

  le_target = *(cl_ulong *)(blk->work->device_target + 24);
  memcpy(clState->cldata, blk->work->midstate, 128);
  for (int i = 0; i < 3; i++)
    ((uint32_t*)clState->cldata)[32 + i] = swab32(data_end[i]);

  status = clEnqueueWriteBuffer(clState->commandQueue, clState->CLbuffer0, true, 0, 128+16, clState->cldata, 0, NULL, NULL);

  // jh80 + keccak - search()
  kernel = &clState->kernel;
  num = 0;
  CL_SET_ARG(clState->CLbuffer0);
  CL_SET_ARG(clState->padbuffer8);
  // echo - search1()
  kernel = clState->extra_kernels;
  num = 0;
  CL_SET_ARG(clState->padbuffer8);
  CL_SET_ARG(clState->outputBuffer);
  CL_SET_ARG(le_target);

  return status;
}

static cl_int queue_veltor_kernel(struct __clState *clState, struct _dev_blk_ctx *blk, __maybe_unused cl_uint threads)
{
  cl_kernel *kernel;
  unsigned int num;
  cl_ulong le_target;
  cl_int status = 0;

  le_target = *(cl_ulong *)(blk->work->device_target + 24);
  flip80(clState->cldata, blk->work->data);
  status = clEnqueueWriteBuffer(clState->commandQueue, clState->CLbuffer0, true, 0, 80, clState->cldata, 0, NULL, NULL);
  // skein80 search()
  kernel = &clState->kernel;
  num = 0;
  CL_SET_ARG(clState->CLbuffer0);
  CL_SET_ARG(clState->padbuffer8);
  // shavite search1()
  kernel = clState->extra_kernels;
  CL_SET_ARG_0(clState->padbuffer8);
  // shabal search2()
  CL_NEXTKERNEL_SET_ARG_0(clState->padbuffer8);
  // streebog search3()
  num = 0;
  CL_NEXTKERNEL_SET_ARG(clState->padbuffer8);
  CL_SET_ARG(clState->outputBuffer);
  CL_SET_ARG(le_target);

  return status;
}

static cl_int queue_skein2_kernel(struct __clState *clState, struct _dev_blk_ctx *blk, __maybe_unused cl_uint threads)
{
  cl_kernel *kernel;
  unsigned int num;
  cl_ulong le_target;
  cl_int status = 0;

  le_target = *(cl_ulong *)(blk->work->device_target + 24);
  flip80(clState->cldata, blk->work->data);
  status = clEnqueueWriteBuffer(clState->commandQueue, clState->CLbuffer0, true, 0, 80, clState->cldata, 0, NULL, NULL);
  // skein80 search()
  kernel = &clState->kernel;
  num = 0;
  CL_SET_ARG(clState->CLbuffer0);
  CL_SET_ARG(clState->padbuffer8);
  // skein64 search1()
  kernel = clState->extra_kernels;
  num = 0;
  CL_SET_ARG(clState->padbuffer8);
  CL_SET_ARG(clState->outputBuffer);
  CL_SET_ARG(le_target);

  return status;
}

static cl_int queue_skunk_kernel(struct __clState *clState, struct _dev_blk_ctx *blk, __maybe_unused cl_uint threads)
{
  cl_kernel *kernel;
  unsigned int num;
  cl_ulong le_target;
  cl_int status = 0;

  le_target = *(cl_ulong *)(blk->work->device_target + 24);
  flip80(clState->cldata, blk->work->data);
  status = clEnqueueWriteBuffer(clState->commandQueue, clState->CLbuffer0, true, 0, 80, clState->cldata, 0, NULL, NULL);
  // skein search()
  kernel = &clState->kernel;
  num = 0;
  CL_SET_ARG(clState->CLbuffer0);
  CL_SET_ARG(clState->padbuffer8);
  // cubehash search1()
  kernel = clState->extra_kernels;
  CL_SET_ARG_0(clState->padbuffer8);
  // fugue search2()
  CL_NEXTKERNEL_SET_ARG_0(clState->padbuffer8);
  // gost/streebog search3()
  num = 0;
  CL_NEXTKERNEL_SET_ARG(clState->padbuffer8);
  CL_SET_ARG(clState->outputBuffer);
  CL_SET_ARG(le_target);

  return status;
}

static cl_int queue_bitblock_kernel(struct __clState *clState, struct _dev_blk_ctx *blk, __maybe_unused cl_uint threads)
{
  cl_kernel *kernel;
  unsigned int num;
  cl_ulong le_target;
  cl_int status = 0;

  le_target = *(cl_ulong *)(blk->work->device_target + 24);
  flip80(clState->cldata, blk->work->data);
  status = clEnqueueWriteBuffer(clState->commandQueue, clState->CLbuffer0, true, 0, 80, clState->cldata, 0, NULL, NULL);

  // blake - search
  kernel = &clState->kernel;
  num = 0;
  CL_SET_ARG(clState->CLbuffer0);
  CL_SET_ARG(clState->padbuffer8);
  // bmw - search1
  kernel = clState->extra_kernels;
  CL_SET_ARG_0(clState->padbuffer8);
  // groestl - search2
  CL_NEXTKERNEL_SET_ARG_0(clState->padbuffer8);
  // skein - search3
  CL_NEXTKERNEL_SET_ARG_0(clState->padbuffer8);
  // jh - search4
  CL_NEXTKERNEL_SET_ARG_0(clState->padbuffer8);
  // keccak - search5
  CL_NEXTKERNEL_SET_ARG_0(clState->padbuffer8);
  // luffa - search6
  CL_NEXTKERNEL_SET_ARG_0(clState->padbuffer8);
  // cubehash - search7
  CL_NEXTKERNEL_SET_ARG_0(clState->padbuffer8);
  // shavite - search8
  CL_NEXTKERNEL_SET_ARG_0(clState->padbuffer8);
  // simd - search9
  CL_NEXTKERNEL_SET_ARG_0(clState->padbuffer8);
  // echo - search10
  CL_NEXTKERNEL_SET_ARG_0(clState->padbuffer8);
  // hamsi - search11
  CL_NEXTKERNEL_SET_ARG_0(clState->padbuffer8);
  // fugue - search12
  CL_NEXTKERNEL_SET_ARG_0(clState->padbuffer8);
  // hamsi - search11
  CL_NEXTKERNEL_SET_ARG_0(clState->padbuffer8);
  // fugue - search12
  num = 0;
  CL_NEXTKERNEL_SET_ARG(clState->padbuffer8);
  CL_SET_ARG(clState->outputBuffer);
  CL_SET_ARG(le_target);

  return status;
}

static cl_int queue_bitblockold_kernel(struct __clState *clState, struct _dev_blk_ctx *blk, __maybe_unused cl_uint threads)
{
  cl_kernel *kernel;
  unsigned int num;
  cl_ulong le_target;
  cl_int status = 0;

  le_target = *(cl_ulong *)(blk->work->device_target + 24);
  flip80(clState->cldata, blk->work->data);
  status = clEnqueueWriteBuffer(clState->commandQueue, clState->CLbuffer0, true, 0, 80, clState->cldata, 0, NULL, NULL);

  // blake - search
  kernel = &clState->kernel;
  num = 0;
  CL_SET_ARG(clState->CLbuffer0);
  CL_SET_ARG(clState->padbuffer8);
  // bmw - search1
  kernel = clState->extra_kernels;
  CL_SET_ARG_0(clState->padbuffer8);
  // groestl - search2
  CL_NEXTKERNEL_SET_ARG_0(clState->padbuffer8);
  // skein - search3
  CL_NEXTKERNEL_SET_ARG_0(clState->padbuffer8);
  // jh - search4
  CL_NEXTKERNEL_SET_ARG_0(clState->padbuffer8);
  // keccak - search5
  CL_NEXTKERNEL_SET_ARG_0(clState->padbuffer8);
  // luffa - search6
  CL_NEXTKERNEL_SET_ARG_0(clState->padbuffer8);
  // cubehash - search7
  CL_NEXTKERNEL_SET_ARG_0(clState->padbuffer8);
  // shavite - search8
  CL_NEXTKERNEL_SET_ARG_0(clState->padbuffer8);
  // simd - search9
  CL_NEXTKERNEL_SET_ARG_0(clState->padbuffer8);
  // combined echo, hamsi, fugue - shabal - whirlpool - search10
  num = 0;
  CL_NEXTKERNEL_SET_ARG(clState->padbuffer8);
  CL_SET_ARG(clState->outputBuffer);
  CL_SET_ARG(le_target);

  return status;
}


static cl_int queue_marucoin_mod_kernel(struct __clState *clState, struct _dev_blk_ctx *blk, __maybe_unused cl_uint threads)
{
  cl_kernel *kernel;
  unsigned int num;
  cl_ulong le_target;
  cl_int status = 0;

  le_target = *(cl_ulong *)(blk->work->device_target + 24);
  flip80(clState->cldata, blk->work->data);
  status = clEnqueueWriteBuffer(clState->commandQueue, clState->CLbuffer0, true, 0, 80, clState->cldata, 0, NULL, NULL);

  // blake - search
  kernel = &clState->kernel;
  num = 0;
  CL_SET_ARG(clState->CLbuffer0);
  CL_SET_ARG(clState->padbuffer8);
  // bmw - search1
  kernel = clState->extra_kernels;
  CL_SET_ARG_0(clState->padbuffer8);
  // groestl - search2
  CL_NEXTKERNEL_SET_ARG_0(clState->padbuffer8);
  // skein - search3
  CL_NEXTKERNEL_SET_ARG_0(clState->padbuffer8);
  // jh - search4
  CL_NEXTKERNEL_SET_ARG_0(clState->padbuffer8);
  // keccak - search5
  CL_NEXTKERNEL_SET_ARG_0(clState->padbuffer8);
  // luffa - search6
  CL_NEXTKERNEL_SET_ARG_0(clState->padbuffer8);
  // cubehash - search7
  CL_NEXTKERNEL_SET_ARG_0(clState->padbuffer8);
  // shavite - search8
  CL_NEXTKERNEL_SET_ARG_0(clState->padbuffer8);
  // simd - search9
  CL_NEXTKERNEL_SET_ARG_0(clState->padbuffer8);
  // echo - search10
  CL_NEXTKERNEL_SET_ARG_0(clState->padbuffer8);
  // hamsi - search11
  CL_NEXTKERNEL_SET_ARG_0(clState->padbuffer8);
  // fugue - search12
  num = 0;
  CL_NEXTKERNEL_SET_ARG(clState->padbuffer8);
  CL_SET_ARG(clState->outputBuffer);
  CL_SET_ARG(le_target);

  return status;
}

static cl_int queue_marucoin_mod_old_kernel(struct __clState *clState, struct _dev_blk_ctx *blk, __maybe_unused cl_uint threads)
{
  cl_kernel *kernel;
  unsigned int num;
  cl_ulong le_target;
  cl_int status = 0;

  le_target = *(cl_ulong *)(blk->work->device_target + 24);
  flip80(clState->cldata, blk->work->data);
  status = clEnqueueWriteBuffer(clState->commandQueue, clState->CLbuffer0, true, 0, 80, clState->cldata, 0, NULL, NULL);

  // blake - search
  kernel = &clState->kernel;
  num = 0;
  CL_SET_ARG(clState->CLbuffer0);
  CL_SET_ARG(clState->padbuffer8);
  // bmw - search1
  kernel = clState->extra_kernels;
  CL_SET_ARG_0(clState->padbuffer8);
  // groestl - search2
  CL_NEXTKERNEL_SET_ARG_0(clState->padbuffer8);
  // skein - search3
  CL_NEXTKERNEL_SET_ARG_0(clState->padbuffer8);
  // jh - search4
  CL_NEXTKERNEL_SET_ARG_0(clState->padbuffer8);
  // keccak - search5
  CL_NEXTKERNEL_SET_ARG_0(clState->padbuffer8);
  // luffa - search6
  CL_NEXTKERNEL_SET_ARG_0(clState->padbuffer8);
  // cubehash - search7
  CL_NEXTKERNEL_SET_ARG_0(clState->padbuffer8);
  // shavite - search8
  CL_NEXTKERNEL_SET_ARG_0(clState->padbuffer8);
  // simd - search9
  CL_NEXTKERNEL_SET_ARG_0(clState->padbuffer8);
  // combined echo, hamsi, fugue - search10
  num = 0;
  CL_NEXTKERNEL_SET_ARG(clState->padbuffer8);
  CL_SET_ARG(clState->outputBuffer);
  CL_SET_ARG(le_target);

  return status;
}

static cl_int queue_talkcoin_mod_kernel(struct __clState *clState, struct _dev_blk_ctx *blk, __maybe_unused cl_uint threads)
{
  cl_kernel *kernel;
  unsigned int num;
  cl_ulong le_target;
  cl_int status = 0;

  le_target = *(cl_ulong *)(blk->work->device_target + 24);
  flip80(clState->cldata, blk->work->data);
  status = clEnqueueWriteBuffer(clState->commandQueue, clState->CLbuffer0, true, 0, 80, clState->cldata, 0, NULL, NULL);

  // blake - search
  kernel = &clState->kernel;
  num = 0;
  CL_SET_ARG(clState->CLbuffer0);
  CL_SET_ARG(clState->padbuffer8);
  // groestl - search1
  kernel = clState->extra_kernels;
  CL_SET_ARG_0(clState->padbuffer8);
  // jh - search2
  CL_NEXTKERNEL_SET_ARG_0(clState->padbuffer8);
  // keccak - search3
  CL_NEXTKERNEL_SET_ARG_0(clState->padbuffer8);
  // skein - search4
  num = 0;
  CL_NEXTKERNEL_SET_ARG(clState->padbuffer8);
  CL_SET_ARG(clState->outputBuffer);
  CL_SET_ARG(le_target);

  return status;
}

static cl_int queue_x14_kernel(struct __clState *clState, struct _dev_blk_ctx *blk, __maybe_unused cl_uint threads)
{
  cl_kernel *kernel;
  unsigned int num;
  cl_ulong le_target;
  cl_int status = 0;

  le_target = *(cl_ulong *)(blk->work->device_target + 24);
  flip80(clState->cldata, blk->work->data);
  status = clEnqueueWriteBuffer(clState->commandQueue, clState->CLbuffer0, true, 0, 80, clState->cldata, 0, NULL, NULL);

  // blake - search
  kernel = &clState->kernel;
  num = 0;
  CL_SET_ARG(clState->CLbuffer0);
  CL_SET_ARG(clState->padbuffer8);
  // bmw - search1
  kernel = clState->extra_kernels;
  CL_SET_ARG_0(clState->padbuffer8);
  // groestl - search2
  CL_NEXTKERNEL_SET_ARG_0(clState->padbuffer8);
  // skein - search3
  CL_NEXTKERNEL_SET_ARG_0(clState->padbuffer8);
  // jh - search4
  CL_NEXTKERNEL_SET_ARG_0(clState->padbuffer8);
  // keccak - search5
  CL_NEXTKERNEL_SET_ARG_0(clState->padbuffer8);
  // luffa - search6
  CL_NEXTKERNEL_SET_ARG_0(clState->padbuffer8);
  // cubehash - search7
  CL_NEXTKERNEL_SET_ARG_0(clState->padbuffer8);
  // shavite - search8
  CL_NEXTKERNEL_SET_ARG_0(clState->padbuffer8);
  // simd - search9
  CL_NEXTKERNEL_SET_ARG_0(clState->padbuffer8);
  // echo - search10
  CL_NEXTKERNEL_SET_ARG_0(clState->padbuffer8);
  // hamsi - search11
  CL_NEXTKERNEL_SET_ARG_0(clState->padbuffer8);
  // fugue - search12
  CL_NEXTKERNEL_SET_ARG_0(clState->padbuffer8);
  // shabal - search13
  num = 0;
  CL_NEXTKERNEL_SET_ARG(clState->padbuffer8);
  CL_SET_ARG(clState->outputBuffer);
  CL_SET_ARG(le_target);

  return status;
}

static cl_int queue_x14_old_kernel(struct __clState *clState, struct _dev_blk_ctx *blk, __maybe_unused cl_uint threads)
{
  cl_kernel *kernel;
  unsigned int num;
  cl_ulong le_target;
  cl_int status = 0;

  le_target = *(cl_ulong *)(blk->work->device_target + 24);
  flip80(clState->cldata, blk->work->data);
  status = clEnqueueWriteBuffer(clState->commandQueue, clState->CLbuffer0, true, 0, 80, clState->cldata, 0, NULL, NULL);

  // blake - search
  kernel = &clState->kernel;
  num = 0;
  CL_SET_ARG(clState->CLbuffer0);
  CL_SET_ARG(clState->padbuffer8);
  // bmw - search1
  kernel = clState->extra_kernels;
  CL_SET_ARG_0(clState->padbuffer8);
  // groestl - search2
  CL_NEXTKERNEL_SET_ARG_0(clState->padbuffer8);
  // skein - search3
  CL_NEXTKERNEL_SET_ARG_0(clState->padbuffer8);
  // jh - search4
  CL_NEXTKERNEL_SET_ARG_0(clState->padbuffer8);
  // keccak - search5
  CL_NEXTKERNEL_SET_ARG_0(clState->padbuffer8);
  // luffa - search6
  CL_NEXTKERNEL_SET_ARG_0(clState->padbuffer8);
  // cubehash - search7
  CL_NEXTKERNEL_SET_ARG_0(clState->padbuffer8);
  // shavite - search8
  CL_NEXTKERNEL_SET_ARG_0(clState->padbuffer8);
  // simd - search9
  CL_NEXTKERNEL_SET_ARG_0(clState->padbuffer8);
  // combined echo, hamsi, fugue - shabal - search10
  num = 0;
  CL_NEXTKERNEL_SET_ARG(clState->padbuffer8);
  CL_SET_ARG(clState->outputBuffer);
  CL_SET_ARG(le_target);

  return status;
}

static cl_int queue_fresh_kernel(struct __clState *clState, struct _dev_blk_ctx *blk, __maybe_unused cl_uint threads)
{
  cl_kernel *kernel;
  unsigned int num;
  cl_ulong le_target;
  cl_int status = 0;

  le_target = *(cl_ulong *)(blk->work->device_target + 24);
  flip80(clState->cldata, blk->work->data);
  status = clEnqueueWriteBuffer(clState->commandQueue, clState->CLbuffer0, true, 0, 80, clState->cldata, 0, NULL, NULL);

  // shavite 1 - search
  kernel = &clState->kernel;
  num = 0;
  CL_SET_ARG(clState->CLbuffer0);
  CL_SET_ARG(clState->padbuffer8);
  // smid 1 - search1
  kernel = clState->extra_kernels;
  CL_SET_ARG_0(clState->padbuffer8);
  // shavite 2 - search2
  CL_NEXTKERNEL_SET_ARG_0(clState->padbuffer8);
  // smid 2 - search3
  CL_NEXTKERNEL_SET_ARG_0(clState->padbuffer8);
  // echo - search4
  num = 0;
  CL_NEXTKERNEL_SET_ARG(clState->padbuffer8);
  CL_SET_ARG(clState->outputBuffer);
  CL_SET_ARG(le_target);

  return status;
}

static cl_int queue_whirlcoin_kernel(struct __clState *clState, struct _dev_blk_ctx *blk, __maybe_unused cl_uint threads)
{
  cl_kernel *kernel;
  cl_ulong le_target;
  cl_int status = 0;

  le_target = *(cl_ulong *)(blk->work->device_target + 24);
  flip80(clState->cldata, blk->work->data);
  status = clEnqueueWriteBuffer(clState->commandQueue, clState->CLbuffer0, true, 0, 80, clState->cldata, 0, NULL, NULL);

  //clbuffer, hashes
  kernel = &clState->kernel;
  CL_SET_ARG_N(0, clState->CLbuffer0);
  CL_SET_ARG_N(1, clState->padbuffer8);

  kernel = clState->extra_kernels;
  CL_SET_ARG_N(0, clState->padbuffer8);

  CL_NEXTKERNEL_SET_ARG_N(0, clState->padbuffer8);

  //hashes, output, target
  CL_NEXTKERNEL_SET_ARG_N(0, clState->padbuffer8);
  CL_SET_ARG_N(1, clState->outputBuffer);
  CL_SET_ARG_N(2, le_target);

  return status;
}

static cl_int queue_whirlpoolx_kernel(struct __clState *clState, struct _dev_blk_ctx *blk, __maybe_unused cl_uint threads)
{
  uint64_t midblock[8], key[8] = { 0 }, tmp[8] = { 0 };
  cl_ulong le_target;
  cl_int status;

  le_target = *(cl_ulong *)(blk->work->device_target + 24);
  flip80(clState->cldata, blk->work->data);

  memcpy(midblock, clState->cldata, 64);

  // midblock = n, key = h
  for (int i = 0; i < 10; ++i) {
    tmp[0] = WHIRLPOOL_ROUND_CONSTANTS[i];
    whirlpool_round(key, tmp);
    tmp[0] = 0;
    whirlpool_round(midblock, tmp);

    for (int x = 0; x < 8; ++x) {
      midblock[x] ^= key[x];
    }
  }

  for (int i = 0; i < 8; ++i) {
    midblock[i] ^= ((uint64_t *)(clState->cldata))[i];
  }

  status = clSetKernelArg(clState->kernel, 0, sizeof(cl_ulong8), (cl_ulong8 *)&midblock);
  status |= clSetKernelArg(clState->kernel, 1, sizeof(cl_ulong), (void *)(((uint64_t *)clState->cldata) + 8));
  status |= clSetKernelArg(clState->kernel, 2, sizeof(cl_ulong), (void *)(((uint64_t *)clState->cldata) + 9));
  status |= clSetKernelArg(clState->kernel, 3, sizeof(cl_mem), (void *)&clState->outputBuffer);
  status |= clSetKernelArg(clState->kernel, 4, sizeof(cl_ulong), (void *)&le_target);

  return status;
}

static cl_int queue_lyra2re_kernel(struct __clState *clState, struct _dev_blk_ctx *blk, __maybe_unused cl_uint threads)
{
  cl_kernel *kernel;
  unsigned int num;
  cl_int status = 0;
  cl_ulong le_target;

  le_target = *(cl_ulong *)(blk->work->device_target + 24);
  flip80(clState->cldata, blk->work->data);
  status = clEnqueueWriteBuffer(clState->commandQueue, clState->CLbuffer0, true, 0, 80, clState->cldata, 0, NULL, NULL);

  // blake - search
  kernel = &clState->kernel;
  num = 0;

  CL_SET_ARG(clState->padbuffer8);
  CL_SET_ARG(blk->work->blk.ctx_a);
  CL_SET_ARG(blk->work->blk.ctx_b);
  CL_SET_ARG(blk->work->blk.ctx_c);
  CL_SET_ARG(blk->work->blk.ctx_d);
  CL_SET_ARG(blk->work->blk.ctx_e);
  CL_SET_ARG(blk->work->blk.ctx_f);
  CL_SET_ARG(blk->work->blk.ctx_g);
  CL_SET_ARG(blk->work->blk.ctx_h);
  CL_SET_ARG(blk->work->blk.cty_a);
  CL_SET_ARG(blk->work->blk.cty_b);
  CL_SET_ARG(blk->work->blk.cty_c);

  // bmw - search1
  kernel = clState->extra_kernels;
  CL_SET_ARG_0(clState->padbuffer8);
  // groestl - search2
  CL_NEXTKERNEL_SET_ARG_0(clState->padbuffer8);
  // skein - search3
  CL_NEXTKERNEL_SET_ARG_0(clState->padbuffer8);
  // jh - search4
  num = 0;
  CL_NEXTKERNEL_SET_ARG(clState->padbuffer8);
  CL_SET_ARG(clState->outputBuffer);
  CL_SET_ARG(le_target);

  return status;
}

static cl_int queue_lyra2rev2_kernel(struct __clState *clState, struct _dev_blk_ctx *blk, __maybe_unused cl_uint threads)
{
  cl_kernel *kernel;
  unsigned int num;
  cl_int status = 0;
  cl_ulong le_target;

  //  le_target = *(cl_uint *)(blk->work->device_target + 28);
  le_target = *(cl_ulong *)(blk->work->device_target + 24);
  flip80(clState->cldata, blk->work->data);
  status = clEnqueueWriteBuffer(clState->commandQueue, clState->CLbuffer0, true, 0, 80, clState->cldata, 0, NULL, NULL);

  // blake - search
  kernel = &clState->kernel;
  num = 0;
  //  CL_SET_ARG(clState->CLbuffer0);
  CL_SET_ARG(clState->buffer1);
  CL_SET_ARG(blk->work->blk.ctx_a);
  CL_SET_ARG(blk->work->blk.ctx_b);
  CL_SET_ARG(blk->work->blk.ctx_c);
  CL_SET_ARG(blk->work->blk.ctx_d);
  CL_SET_ARG(blk->work->blk.ctx_e);
  CL_SET_ARG(blk->work->blk.ctx_f);
  CL_SET_ARG(blk->work->blk.ctx_g);
  CL_SET_ARG(blk->work->blk.ctx_h);
  CL_SET_ARG(blk->work->blk.cty_a);
  CL_SET_ARG(blk->work->blk.cty_b);
  CL_SET_ARG(blk->work->blk.cty_c);

  // keccak - search1
  kernel = clState->extra_kernels;
  CL_SET_ARG_0(clState->buffer1);
  // cubehash - search2
  num = 0;
  CL_NEXTKERNEL_SET_ARG_0(clState->buffer1);
  // lyra - search3
  num = 0;
  CL_NEXTKERNEL_SET_ARG_N(0, clState->buffer1);
  CL_SET_ARG_N(1, clState->padbuffer8);
  // skein -search4
  num = 0;
  CL_NEXTKERNEL_SET_ARG_0(clState->buffer1);
  // cubehash - search5
  num = 0;
  CL_NEXTKERNEL_SET_ARG_0(clState->buffer1);
  // bmw - search6
  num = 0;
  CL_NEXTKERNEL_SET_ARG(clState->buffer1);
  CL_SET_ARG(clState->outputBuffer);
  CL_SET_ARG(le_target);

  return status;
}

static cl_int queue_pluck_kernel(_clState *clState, dev_blk_ctx *blk, __maybe_unused cl_uint threads)
{
  cl_kernel *kernel = &clState->kernel;
  unsigned int num = 0;
  cl_uint le_target;
  cl_int status = 0;

  le_target = (cl_uint)le32toh(((uint32_t *)blk->work->/*device_*/target)[7]);
  flip80(clState->cldata, blk->work->data);
  status = clEnqueueWriteBuffer(clState->commandQueue, clState->CLbuffer0, true, 0, 80, clState->cldata, 0, NULL, NULL);

  CL_SET_ARG(clState->CLbuffer0);
  CL_SET_ARG(clState->outputBuffer);
  CL_SET_ARG(clState->padbuffer8);
  CL_SET_ARG(le_target);

  return status;
}

static cl_int queue_blake_kernel(_clState *clState, dev_blk_ctx *blk, __maybe_unused cl_uint threads)
{
  cl_kernel *kernel = &clState->kernel;
  unsigned int num = 0;
  cl_int status = 0;
  cl_ulong le_target;

  le_target = *(cl_ulong *)(blk->work->device_target + 24);
  flip80(clState->cldata, blk->work->data);
  status = clEnqueueWriteBuffer(clState->commandQueue, clState->CLbuffer0, true, 0, 80, clState->cldata, 0, NULL, NULL);

  CL_SET_ARG(clState->outputBuffer);
  CL_SET_ARG(blk->work->blk.ctx_a);
  CL_SET_ARG(blk->work->blk.ctx_b);
  CL_SET_ARG(blk->work->blk.ctx_c);
  CL_SET_ARG(blk->work->blk.ctx_d);
  CL_SET_ARG(blk->work->blk.ctx_e);
  CL_SET_ARG(blk->work->blk.ctx_f);
  CL_SET_ARG(blk->work->blk.ctx_g);
  CL_SET_ARG(blk->work->blk.ctx_h);

  CL_SET_ARG(blk->work->blk.cty_a);
  CL_SET_ARG(blk->work->blk.cty_b);
  CL_SET_ARG(blk->work->blk.cty_c);

  return status;
}

static cl_int queue_decred_kernel(_clState *clState, dev_blk_ctx *blk, __maybe_unused cl_uint threads)
{
  cl_kernel *kernel = &clState->kernel;
  unsigned int num = 0;
  cl_int status = 0;

  CL_SET_ARG(clState->outputBuffer);
  /* Midstate */
  CL_SET_BLKARG(ctx_a);
  CL_SET_BLKARG(ctx_b);
  CL_SET_BLKARG(ctx_c);
  CL_SET_BLKARG(ctx_d);
  CL_SET_BLKARG(ctx_e);
  CL_SET_BLKARG(ctx_f);
  CL_SET_BLKARG(ctx_g);
  CL_SET_BLKARG(ctx_h);
  /* Last 52 bytes of data (without nonce) */
  CL_SET_BLKARG(cty_a);
  CL_SET_BLKARG(cty_b);
  CL_SET_BLKARG(cty_c);
  CL_SET_BLKARG(cty_d);
  CL_SET_BLKARG(cty_e);
  CL_SET_BLKARG(cty_f);
  CL_SET_BLKARG(cty_g);
  CL_SET_BLKARG(cty_h);
  CL_SET_BLKARG(cty_i);
  CL_SET_BLKARG(cty_j);
  CL_SET_BLKARG(cty_k);
  CL_SET_BLKARG(cty_l);

  return status;
}

extern cglock_t EthCacheLock[2];
extern uint8_t* EthCache[2];

extern pthread_mutex_t eth_nonce_lock;
extern uint32_t eth_nonce;


static cl_int queue_ethash_kernel(_clState *clState, dev_blk_ctx *blk, __maybe_unused cl_uint threads)
{
  cl_kernel *kernel;
  unsigned int num = 0;
  cl_int status = 0;
  cl_ulong le_target;
  cl_uint HighNonce, Isolate = 0xFFFFFFFFUL;
  cl_ulong DAGSize = EthGetDAGSize(blk->work->EpochNumber);
  size_t DAGItems = (size_t) (DAGSize / 64);
  struct cgpu_info *cgpu = blk->work->thr ? blk->work->thr->cgpu : NULL; // see get_work()

  le_target = *(cl_ulong *)(blk->work->device_target + 24);

  // DO NOT flip80.
  status = clEnqueueWriteBuffer(clState->commandQueue, clState->CLbuffer0, true, 0, 32, blk->work->data, 0, NULL, NULL);
  if (clState->EpochNumber != blk->work->EpochNumber)
  {
    clState->EpochNumber = blk->work->EpochNumber;
    cl_ulong CacheSize = EthGetCacheSize(blk->work->EpochNumber);
    cl_event DAGGenEvent;

    applog(LOG_INFO, "DAG being regenerated on %s", cgpu->name);
    if (clState->EthCache)
      clReleaseMemObject(clState->EthCache);
    if (clState->DAG)
      clReleaseMemObject(clState->DAG);

    clState->DAG = clCreateBuffer(clState->context, CL_MEM_READ_WRITE, DAGSize, NULL, &status);
    if (status != CL_SUCCESS) {
      applog(LOG_ERR, "Error %d: Creating the DAG buffer.", status);
      return(status);
    }

    clState->EthCache = clCreateBuffer(clState->context, CL_MEM_READ_ONLY, CacheSize, NULL, &status);

    int idx = blk->work->EpochNumber % 2;
    cg_ilock(&EthCacheLock[idx]);
    bool update = (EthCache[idx] == NULL || *(uint32_t*) EthCache[idx] != blk->work->EpochNumber);
    if (update) {
      cg_ulock(&EthCacheLock[idx]);
      EthCache[idx] = (uint8_t*) realloc(EthCache[idx], sizeof(uint8_t) * CacheSize + 64);
      *(uint32_t*) EthCache[idx] = blk->work->EpochNumber;
      EthGenerateCache(EthCache[idx] + 64, blk->work->seedhash, CacheSize);
    }
    else
      cg_dlock(&EthCacheLock[idx]);

    if (status == CL_SUCCESS)
      status = clEnqueueWriteBuffer(clState->commandQueue, clState->EthCache, true, 0, sizeof(cl_uchar) * CacheSize, EthCache[idx] + 64, 0, NULL, NULL);

    if (update)
      cg_wunlock(&EthCacheLock[idx]);
    else
      cg_runlock(&EthCacheLock[idx]);

    if (status != CL_SUCCESS) {
      applog(LOG_ERR, "Error %d: Creating the cache buffer and/or writing to it.", status);
      return(status);
    }

    cl_uint zero = 0;
    cl_uint CacheSize64 = CacheSize / 64;

    size_t items = 1UL << 21;

    // Enqueue DAG gen kernel (multiple launches to prevent driver locks)
    kernel = &clState->GenerateDAG;

    for(size_t p = 0; status == 0 && p < DAGItems/items; p++)
    {
      zero = (cl_uint) p * items;
      num = 0;
      CL_SET_ARG(zero);
      CL_SET_ARG(clState->EthCache);
      CL_SET_ARG(clState->DAG);
      CL_SET_ARG(CacheSize64);
      CL_SET_ARG(Isolate);
      status |= clEnqueueNDRangeKernel(clState->commandQueue, clState->GenerateDAG, 1, NULL, &items, NULL, 0, NULL, &DAGGenEvent);
      status |= clWaitForEvents(1, &DAGGenEvent);
      applog(LOG_INFO, "Generating DAG %s %2.0f%%", cgpu->name, ((double)(zero+items) / DAGItems) * 100);
    }

    // Last items..
    if (status == 0 && DAGItems % items) {
      items = DAGItems % items;
      zero = (cl_uint) DAGItems - items;
      num = 0;
      CL_SET_ARG(zero);
      CL_SET_ARG(clState->EthCache);
      CL_SET_ARG(clState->DAG);
      CL_SET_ARG(CacheSize64);
      CL_SET_ARG(Isolate);
      status |= clEnqueueNDRangeKernel(clState->commandQueue, clState->GenerateDAG, 1, NULL, &items, NULL, 0, NULL, &DAGGenEvent);
      status |= clWaitForEvents(1, &DAGGenEvent);
    }

    clReleaseEvent(DAGGenEvent);

    if(status != CL_SUCCESS) {
      applog(LOG_ERR, "Error %d: Setting args for the DAG kernel and/or executing it.", status);
      return(status);
    }
    applog(LOG_NOTICE, "DAG ready on %s (%u MB)", cgpu->name, (unsigned) (DAGSize >> 20));
  }

  mutex_lock(&eth_nonce_lock);
  HighNonce = eth_nonce++;
  blk->work->Nonce = (cl_ulong) HighNonce << 32;
  mutex_unlock(&eth_nonce_lock);

  num = 0;
  kernel = &clState->kernel;

  // Not nodes now (64 bytes), but DAG entries (128 bytes)
  cl_uint ItemsArg = DAGItems >> 1;

  CL_SET_ARG(clState->outputBuffer);
  CL_SET_ARG(clState->CLbuffer0);
  CL_SET_ARG(clState->DAG);
  CL_SET_ARG(ItemsArg);
  CL_SET_ARG(blk->work->Nonce);
  CL_SET_ARG(le_target);
  CL_SET_ARG(Isolate);

  return(status);
}

// TOFIX: properly adjust to nightcap
static cl_int queue_nightcap_kernel(_clState *clState, dev_blk_ctx *blk, __maybe_unused cl_uint threads)
{
  cl_kernel *kernel;
  unsigned int num = 0;
  cl_int status = 0;
  cl_uint le_target;
  cl_ulong DAGSize = nightcap_get_full_size(blk->work->HeightNumber); //EthGetDAGSize(blk->work->EpochNumber);
  size_t DAGItems = (size_t) (DAGSize / sizeof(NightcapNode));
  struct cgpu_info *cgpu = blk->work->thr ? blk->work->thr->cgpu : NULL; // see get_work()

  assert(sizeof(NightcapNode) == 32);

  le_target = *(cl_ulong *)(blk->work->device_target + 28); // target we should be looking for in kernel


  // Submit block header data
  status = clEnqueueWriteBuffer(clState->commandQueue, clState->CLbuffer0, true, 0, 80, blk->work->data, 0, NULL, NULL);

  // Regenerate the dag if neccesary
  if (clState->EpochNumber != blk->work->EpochNumber)
  {
    clState->EpochNumber = blk->work->EpochNumber;
    cl_ulong CacheSize = nightcap_get_cache_size(blk->work->HeightNumber);  //NightcapGetCacheSize(blk->work->EpochNumber);
    cl_event DAGGenEvent;

    applog(LOG_INFO, "DAG being regenerated on %s", cgpu->name);
    if (clState->EthCache)
      clReleaseMemObject(clState->EthCache);
    if (clState->DAG)
      clReleaseMemObject(clState->DAG);

    clState->DAG = clCreateBuffer(clState->context, CL_MEM_READ_WRITE, DAGSize, NULL, &status);
    if (status != CL_SUCCESS) {
      applog(LOG_ERR, "Error %d: Creating the DAG buffer.", status);
      return(status);
    }

    clState->EthCache = clCreateBuffer(clState->context, CL_MEM_READ_ONLY, CacheSize, NULL, &status);

    int idx = blk->work->EpochNumber % 2;
    cg_ilock(&EthCacheLock[idx]);
    bool update = (EthCache[idx] == NULL || *(uint32_t*) EthCache[idx] != blk->work->EpochNumber);
    if (update) {
      cg_ulock(&EthCacheLock[idx]);
      EthCache[idx] = (uint8_t*) realloc(EthCache[idx], sizeof(uint8_t) * CacheSize + 64); // NOTE: for some reason at least 64 bytes
      *(uint32_t*) EthCache[idx] = blk->work->EpochNumber;
		  NightcapGenerateCache(EthCache[idx] + 64, blk->work->seedhash, CacheSize);           // Cache is offset by 64 again
		  /*
      FILE* fp = fopen("nightcap_cache.dat", "wb");
      fwrite(EthCache[idx] + 64, 1, CacheSize, fp);
      fclose(fp);
		exit(1);*/
    }
    else
      cg_dlock(&EthCacheLock[idx]);

    if (status == CL_SUCCESS) {
      // Load cache, offset by 64 again
      status = clEnqueueWriteBuffer(clState->commandQueue, clState->EthCache, true, 0, sizeof(cl_uchar) * CacheSize, EthCache[idx] + 64, 0, NULL, NULL);
    }

    if (update)
      cg_wunlock(&EthCacheLock[idx]);
    else
      cg_runlock(&EthCacheLock[idx]);

    if (status != CL_SUCCESS) {
      applog(LOG_ERR, "Error %d: Creating the cache buffer and/or writing to it.", status);
      return(status);
    }

    cl_uint zero = 0;
    cl_uint CacheSizeNodes = CacheSize / sizeof(NightcapNode);

    size_t items = 1UL << 21;  // NOTE: this is the work unit we are using for dag

    // Enqueue DAG gen kernel (multiple launches to prevent driver locks)
    kernel = &clState->GenerateDAG;

    for(size_t p = 0; status == 0 && p < DAGItems/items; p++)
    {
      // Set offset for our full items size workload
      zero = (cl_uint) p * items;
      num = 0;
      CL_SET_ARG(zero);
      CL_SET_ARG(clState->EthCache);
      CL_SET_ARG(clState->DAG);
      CL_SET_ARG(CacheSizeNodes);
      //CL_SET_ARG(Isolate);
      status |= clEnqueueNDRangeKernel(clState->commandQueue, clState->GenerateDAG, 1, NULL, &items, NULL, 0, NULL, &DAGGenEvent);
      status |= clWaitForEvents(1, &DAGGenEvent);
      applog(LOG_INFO, "Generating DAG %s %2.0f%%", cgpu->name, ((double)(zero+items) / DAGItems) * 100);
    }

    // Last items..
    if (status == 0 && DAGItems % items) {
      // Work on our last items
      // NOTE: could be simplified by just using one loop
      items = DAGItems % items;
      zero = (cl_uint) DAGItems - items;
      num = 0;
      CL_SET_ARG(zero);
      CL_SET_ARG(clState->EthCache);
      CL_SET_ARG(clState->DAG);
      CL_SET_ARG(CacheSizeNodes);
      //CL_SET_ARG(Isolate);
      status |= clEnqueueNDRangeKernel(clState->commandQueue, clState->GenerateDAG, 1, NULL, &items, NULL, 0, NULL, &DAGGenEvent);
      status |= clWaitForEvents(1, &DAGGenEvent);
    }

    clReleaseEvent(DAGGenEvent);

    if(status != CL_SUCCESS) {
      applog(LOG_ERR, "Error %d: Setting args for the DAG kernel and/or executing it.", status);
      return(status);
    }
    applog(LOG_NOTICE, "DAG ready on %s (%u MB)", cgpu->name, (unsigned) (DAGSize >> 20));
    //exit(0); // DEBUG
  }

  num = 0;
  kernel = &clState->kernel;

  // Not nodes now (32 bytes), but DAG entries (64 bytes)
  cl_uint ItemsArg = DAGItems >> 1;

  CL_SET_ARG(clState->outputBuffer);
  CL_SET_ARG(clState->CLbuffer0);
  CL_SET_ARG(clState->DAG);
  CL_SET_ARG(ItemsArg);
  CL_SET_ARG(le_target);

  return(status);
}

static algorithm_settings_t algos[] = {
  // kernels starting from this will have difficulty calculated by using litecoin algorithm
#define A_SCRYPT(a) \
  { a, ALGO_SCRYPT, "", 1, 65536, 65536, 0, 0, 0xFF, 0xFFFFFFFFULL, 0x0000ffffUL, 0, -1, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, scrypt_regenhash, NULL, queue_scrypt_kernel, gen_hash, append_scrypt_compiler_options }
  A_SCRYPT("ckolivas"),
  A_SCRYPT("alexkarnew"),
  A_SCRYPT("alexkarnold"),
  A_SCRYPT("bufius"),
  A_SCRYPT("psw"),
  A_SCRYPT("zuikkis"),
  A_SCRYPT("arebyp"),
#undef A_SCRYPT

#define A_NEOSCRYPT(a) \
  { a, ALGO_NEOSCRYPT, "", 1, 65536, 65536, 0, 0, 0xFF, 0xFFFF000000000000ULL, 0x0000ffffUL, 0, -1, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, neoscrypt_regenhash, NULL, queue_neoscrypt_kernel, gen_hash, append_neoscrypt_compiler_options }
  A_NEOSCRYPT("neoscrypt"),
#undef A_NEOSCRYPT

#define A_PLUCK(a) \
  { a, ALGO_PLUCK, "", 1, 65536, 65536, 0, 0, 0xFF, 0xFFFF000000000000ULL, 0x0000ffffUL, 0, -1, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, pluck_regenhash, NULL, queue_pluck_kernel, gen_hash, append_neoscrypt_compiler_options }
  A_PLUCK("pluck"),
#undef A_PLUCK

#define A_CREDITS(a) \
  { a, ALGO_CRE, "", 1, 1, 1, 0, 0, 0xFF, 0xFFFF000000000000ULL, 0x0000ffffUL, 0, -1, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, credits_regenhash, NULL, queue_credits_kernel, gen_hash, NULL}
  A_CREDITS("credits"),
#undef A_CREDITS

#define A_DECRED(a) \
  { a, ALGO_DECRED, "", 1, 1, 1, 0, 0, 0xFF, 0xFFFFULL, 0x0, 0, 0, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, decred_regenhash, precalc_hash_decred, queue_decred_kernel, raw_merkle, append_blake256_compiler_options }
  A_DECRED("decred"),
#undef A_DECRED

#ifndef _MSC_VER
#define A_YESCRYPT(a) \
  { a, ALGO_YESCRYPT, "", 1, 65536, 65536, 0, 0, 0xFF, 0xFFFF000000000000ULL, 0x0000ffffUL, 0, -1, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, yescrypt_regenhash, NULL, queue_yescrypt_kernel, gen_hash, append_neoscrypt_compiler_options}
  A_YESCRYPT("yescrypt"),
#undef A_YESCRYPT

#define A_YESCRYPT_MULTI(a) \
  { a, ALGO_YESCRYPT_MULTI, "", 1, 65536, 65536, 0, 0, 0xFF, 0x00000000FFFFULL, 0x0000ffffUL, 6,-1,CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE , yescrypt_regenhash, NULL, queue_yescrypt_multikernel, gen_hash, append_neoscrypt_compiler_options}
  A_YESCRYPT_MULTI("yescrypt-multi"),
#undef A_YESCRYPT_MULTI
#endif /* _MSC_VER yescript code to fix, not my job... */

  // kernels starting from this will have difficulty calculated by using quarkcoin algorithm
#define A_QUARK(a, b) \
  { a, ALGO_QUARK, "", 256, 256, 256, 0, 0, 0xFF, 0xFFFFFFULL, 0x0000ffffUL, 0, 0, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, b, NULL, queue_sph_kernel, gen_hash, append_x11_compiler_options }
  A_QUARK("quarkcoin", quarkcoin_regenhash),
  A_QUARK("qubitcoin", qubitcoin_regenhash),
  A_QUARK("animecoin", animecoin_regenhash),
  A_QUARK("sifcoin", sifcoin_regenhash),
#undef A_QUARK

  // kernels starting from this will have difficulty calculated by using bitcoin algorithm
#define A_DARK(a, b) \
  { a, ALGO_X11, "", 1, 1, 1, 0, 0, 0xFF, 0xFFFFULL, 0x0000ffffUL, 0, 0, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, b, NULL, queue_sph_kernel, gen_hash, append_x11_compiler_options }
  A_DARK("darkcoin", darkcoin_regenhash),
  A_DARK("inkcoin", inkcoin_regenhash),
  A_DARK("myriadcoin-groestl", myriadcoin_groestl_regenhash),
#undef A_DARK

  { "twecoin", ALGO_TWE, "", 1, 1, 1, 0, 0, 0xFF, 0xFFFFULL, 0x0000ffffUL, 0, 0, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, twecoin_regenhash, NULL, queue_sph_kernel, sha256, NULL },
  { "maxcoin", ALGO_KECCAK, "", 1, 256, 1, 4, 15, 0x0F, 0xFFFFULL, 0x000000ffUL, 0, 0, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, maxcoin_regenhash, NULL, queue_maxcoin_kernel, sha256, NULL },

  { "darkcoin-mod", ALGO_X11, "", 1, 1, 1, 0, 0, 0xFF, 0xFFFFULL, 0x0000ffffUL, 10, 8 * 16 * 4194304, 0, darkcoin_regenhash, NULL, queue_darkcoin_mod_kernel, gen_hash, append_x11_compiler_options },

  { "sibcoin-mod", ALGO_SIBCOIN, "", 1, 1, 1, 0, 0, 0xFF, 0xFFFFULL, 0x0000ffffUL, 11, 8 * 16 * 4194304, 0, sibcoin_regenhash, NULL, queue_sibcoin_kernel, gen_hash, append_x11_compiler_options },

  { "skein2", ALGO_SKEIN2, "", 1, 1, 1, 0, 0, 0xFF, 0xFFFFULL, 0x0000ffffUL, 1, 8 * 16 * 4194304, 0, skein2_regenhash, NULL, queue_skein2_kernel, gen_hash, append_x11_compiler_options },

  { "skunk", ALGO_SKUNK, "", 1, 1, 1, 0, 0, 0xFF, 0xFFFFULL, 0x0000ffffUL, 3, 8 * 16 * 4194304, 0, skunk_regenhash, precalc_hash_skunk, queue_skunk_kernel, gen_hash, append_x11_compiler_options },

  { "tribus", ALGO_TRIBUS, "", 1, 1, 1, 0, 0, 0xFF, 0xFFFFULL, 0x0000ffffUL, 1, 4 * 16 * 4194304, 0, tribus_regenhash, precalc_hash_tribus, queue_tribus_kernel, gen_hash, append_x11_compiler_options },

  { "veltor", ALGO_VELTOR, "", 1, 1, 1, 0, 0, 0xFF, 0xFFFFULL, 0x0000ffffUL, 3, 8 * 16 * 4194304, 0, veltor_regenhash, NULL, queue_veltor_kernel, gen_hash, append_x11_compiler_options },

  { "marucoin", ALGO_X13, "", 1, 1, 1, 0, 0, 0xFF, 0xFFFFULL, 0x0000ffffUL, 0, 0, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, marucoin_regenhash, NULL, queue_sph_kernel, gen_hash, append_x13_compiler_options },
  { "marucoin-mod", ALGO_X13, "", 1, 1, 1, 0, 0, 0xFF, 0xFFFFULL, 0x0000ffffUL, 12, 8 * 16 * 4194304, 0, marucoin_regenhash, NULL, queue_marucoin_mod_kernel, gen_hash, append_x13_compiler_options },
  { "marucoin-modold", ALGO_X13, "", 1, 1, 1, 0, 0, 0xFF, 0xFFFFULL, 0x0000ffffUL, 10, 8 * 16 * 4194304, 0, marucoin_regenhash, NULL, queue_marucoin_mod_old_kernel, gen_hash, append_x13_compiler_options },

  { "x14", ALGO_X14, "", 1, 1, 1, 0, 0, 0xFF, 0xFFFFULL, 0x0000ffffUL, 13, 8 * 16 * 4194304, 0, x14_regenhash, NULL, queue_x14_kernel, gen_hash, append_x13_compiler_options },
  { "x14old", ALGO_X14, "", 1, 1, 1, 0, 0, 0xFF, 0xFFFFULL, 0x0000ffffUL, 10, 8 * 16 * 4194304, 0, x14_regenhash, NULL, queue_x14_old_kernel, gen_hash, append_x13_compiler_options },

  { "bitblock", ALGO_X15, "", 1, 1, 1, 0, 0, 0xFF, 0xFFFFULL, 0x0000ffffUL, 14, 4 * 16 * 4194304, 0, bitblock_regenhash, NULL, queue_bitblock_kernel, gen_hash, append_x13_compiler_options },
  { "bitblockold", ALGO_X15, "", 1, 1, 1, 0, 0, 0xFF, 0xFFFFULL, 0x0000ffffUL, 10, 4 * 16 * 4194304, 0, bitblock_regenhash, NULL, queue_bitblockold_kernel, gen_hash, append_x13_compiler_options },

  { "talkcoin-mod", ALGO_NIST, "", 1, 1, 1, 0, 0, 0xFF, 0xFFFFULL, 0x0000ffffUL, 4, 8 * 16 * 4194304, 0, talkcoin_regenhash, NULL, queue_talkcoin_mod_kernel, gen_hash, append_x11_compiler_options },

  { "fresh", ALGO_FRESH, "", 1, 256, 256, 0, 0, 0xFF, 0xFFFFULL, 0x0000ffffUL, 4, 4 * 16 * 4194304, 0, fresh_regenhash, NULL, queue_fresh_kernel, gen_hash, NULL },

  { "lyra2re", ALGO_LYRA2RE, "", 1, 128, 128, 0, 0, 0xFF, 0xFFFFULL, 0x0000ffffUL, 4, 2 * 8 * 4194304, 0, lyra2re_regenhash, precalc_hash_blake256, queue_lyra2re_kernel, gen_hash, NULL },
  { "lyra2rev2", ALGO_LYRA2REV2, "", 1, 256, 256, 0, 0, 0xFF, 0xFFFFULL, 0x0000ffffUL, 6, -1, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, lyra2rev2_regenhash, precalc_hash_blake256, queue_lyra2rev2_kernel, gen_hash, append_neoscrypt_compiler_options },

  // kernels starting from this will have difficulty calculated by using fuguecoin algorithm
#define A_FUGUE(a, b, c) \
  { a, ALGO_FUGUE, "", 1, 256, 256, 0, 0, 0xFF, 0xFFFFULL, 0x0000ffffUL, 0, 0, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, b, NULL, queue_sph_kernel, c, NULL }
  A_FUGUE("fuguecoin", fuguecoin_regenhash, sha256),
  A_FUGUE("groestlcoin", groestlcoin_regenhash, sha256),
  A_FUGUE("diamond", groestlcoin_regenhash, gen_hash),
#undef A_FUGUE

  { "whirlcoin", ALGO_WHIRL, "", 1, 1, 1, 0, 0, 0xFF, 0xFFFFULL, 0x0000ffffUL, 3, 8 * 16 * 4194304, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, whirlcoin_regenhash, NULL, queue_whirlcoin_kernel, sha256, NULL },
  { "whirlpoolx", ALGO_WHIRLPOOLX, "", 1, 1, 1, 0, 0, 0xFF, 0xFFFFULL, 0x0000FFFFUL, 0, 0, 0, whirlpoolx_regenhash, NULL, queue_whirlpoolx_kernel, gen_hash, NULL },

  { "blake256r8",  ALGO_BLAKECOIN, "", 1, 1, 1, 0, 0, 0xFF, 0xFFFFULL, 0x000000ffUL, 0, 128, 0, blakecoin_regenhash, precalc_hash_blakecoin, queue_blake_kernel, sha256,   NULL },
  { "blake256r14", ALGO_BLAKE,     "", 1, 1, 1, 0, 0, 0xFF, 0xFFFFULL, 0x00000000UL, 0, 128, 0, blake256_regenhash, precalc_hash_blake256, queue_blake_kernel, gen_hash, NULL },
  { "vanilla",     ALGO_VANILLA,   "", 1, 1, 1, 0, 0, 0xFF, 0xFFFFULL, 0x000000ffUL, 0, 128, 0, blakecoin_regenhash, precalc_hash_blakecoin, queue_blake_kernel, gen_hash, NULL },

  { "ethash",     ALGO_ETHASH,   "", (1ULL << 32), (1ULL << 32), 1, 0, 0, 0xFF, 0xFFFF000000000000ULL, 0x00000000UL, 0, 128, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, ethash_regenhash, NULL, queue_ethash_kernel, gen_hash, append_ethash_compiler_options },
  
  // TODO: use correct params here
  { "nightcap",   ALGO_NIGHTCAP,   "", 65536, 65536, 1, 0, 0, 0xFF, 0xFFFFFFFFULL, 0x0000ffffUL, 0, 128, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, nightcap_regenhash, NULL, queue_nightcap_kernel, gen_hash, append_nightcap_compiler_options },

  // Terminator (do not remove)
  { NULL, ALGO_UNK, "", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, NULL, NULL, NULL, NULL }
};

void copy_algorithm_settings(algorithm_t* dest, const char* algo)
{
  algorithm_settings_t* src;

  // Find algorithm settings and copy
  for (src = algos; src->name; src++)
  {
    if (strcasecmp(src->name, algo) == 0)
    {
      strcpy(dest->name, src->name);
      dest->kernelfile = src->kernelfile;
      dest->type = src->type;

      dest->diff_multiplier1 = src->diff_multiplier1;
      dest->diff_multiplier2 = src->diff_multiplier2;
      dest->share_diff_multiplier = src->share_diff_multiplier;
      dest->xintensity_shift = src->xintensity_shift;
      dest->intensity_shift = src->intensity_shift;
      dest->found_idx = src->found_idx;
      dest->diff_numerator = src->diff_numerator;
      dest->diff1targ = src->diff1targ;
      dest->n_extra_kernels = src->n_extra_kernels;
      dest->rw_buffer_size = src->rw_buffer_size;
      dest->cq_properties = src->cq_properties;
      dest->regenhash = src->regenhash;
      dest->precalc_hash = src->precalc_hash;
      dest->queue_kernel = src->queue_kernel;
      dest->gen_hash = src->gen_hash;
      dest->set_compile_options = src->set_compile_options;
      break;
    }
  }

  // if not found
  if (src->name == NULL)
  {
    applog(LOG_WARNING, "Algorithm %s not found, using %s.", algo, algos->name);
    copy_algorithm_settings(dest, algos->name);
  }
}

static const char *lookup_algorithm_alias(const char *lookup_alias, uint8_t *nfactor)
{
#define ALGO_ALIAS_NF(alias, name, nf) \
  if (strcasecmp(alias, lookup_alias) == 0) { *nfactor = nf; return name; }
#define ALGO_ALIAS(alias, name) \
  if (strcasecmp(alias, lookup_alias) == 0) return name;

  ALGO_ALIAS_NF("scrypt", "ckolivas", 10);
  ALGO_ALIAS_NF("scrypt", "ckolivas", 10);
  ALGO_ALIAS_NF("adaptive-n-factor", "ckolivas", 11);
  ALGO_ALIAS_NF("adaptive-nfactor", "ckolivas", 11);
  ALGO_ALIAS_NF("nscrypt", "ckolivas", 11);
  ALGO_ALIAS_NF("adaptive-nscrypt", "ckolivas", 11);
  ALGO_ALIAS_NF("adaptive-n-scrypt", "ckolivas", 11);
  ALGO_ALIAS("x11mod", "darkcoin-mod");
  ALGO_ALIAS("x11", "darkcoin-mod");
  ALGO_ALIAS("x13mod", "marucoin-mod");
  ALGO_ALIAS("x13", "marucoin-mod");
  ALGO_ALIAS("x13old", "marucoin-modold");
  ALGO_ALIAS("x13modold", "marucoin-modold");
  ALGO_ALIAS("x15mod", "bitblock");
  ALGO_ALIAS("x15", "bitblock");
  ALGO_ALIAS("x15modold", "bitblockold");
  ALGO_ALIAS("x15old", "bitblockold");
  ALGO_ALIAS("nist5", "talkcoin-mod");
  ALGO_ALIAS("keccak", "maxcoin");
  ALGO_ALIAS("whirlpool", "whirlcoin");
  ALGO_ALIAS("lyra2", "lyra2re");
  ALGO_ALIAS("lyra2v2", "lyra2rev2");
  ALGO_ALIAS("blakecoin", "blake256r8");
  ALGO_ALIAS("blake", "blake256r14");
  ALGO_ALIAS("sibcoin-mod", "sibcoin");
  ALGO_ALIAS("sib", "sibcoin");
  ALGO_ALIAS("doubleskein", "skein2");
  ALGO_ALIAS("woodcoin", "skein2");
  ALGO_ALIAS("skunkhash", "skunk");
  ALGO_ALIAS("signatum", "skunk");
  ALGO_ALIAS("thorsriddle", "veltor");

#undef ALGO_ALIAS
#undef ALGO_ALIAS_NF

  return NULL;
}

void set_algorithm(algorithm_t* algo, const char* newname_alias)
{
  const char *newname;

  //load previous algorithm nfactor in case nfactor was applied before algorithm... or default to 10
  uint8_t old_nfactor = ((algo->nfactor) ? algo->nfactor : 0);
  //load previous kernel file name if was applied before algorithm...
  const char *kernelfile = algo->kernelfile;
  uint8_t nfactor = 10;

  if (!(newname = lookup_algorithm_alias(newname_alias, &nfactor)))
    newname = newname_alias;

  copy_algorithm_settings(algo, newname);

  // use old nfactor if it was previously set and is different than the one set by alias
  if ((old_nfactor > 0) && (old_nfactor != nfactor))
    nfactor = old_nfactor;

  set_algorithm_nfactor(algo, nfactor);

  //reapply kernelfile if was set
  if (!empty_string(kernelfile)) {
    algo->kernelfile = kernelfile;
  }
}

void set_algorithm_nfactor(algorithm_t* algo, const uint8_t nfactor)
{
  algo->nfactor = nfactor;
  algo->n = (1 << nfactor);

  //adjust algo type accordingly
  switch (algo->type)
  {
  case ALGO_SCRYPT:
    //if nfactor isnt 10, switch to NSCRYPT
    if (algo->nfactor != 10)
      algo->type = ALGO_NSCRYPT;
    break;
    //nscrypt
  case ALGO_NSCRYPT:
    //if nfactor is 10, switch to SCRYPT
    if (algo->nfactor == 10)
      algo->type = ALGO_SCRYPT;
    break;
    //ignore rest
  default:
    break;
  }
}

bool cmp_algorithm(const algorithm_t* algo1, const algorithm_t* algo2)
{
  return (!safe_cmp(algo1->name, algo2->name) && !safe_cmp(algo1->kernelfile, algo2->kernelfile) && (algo1->nfactor == algo2->nfactor));
}
