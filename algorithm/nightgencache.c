#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "sph/sph_blake.h"

#include "algorithm/nightcap.h"

#include "logging.h"

// Output (cache_nodes) MUST have at least cache_size bytes
void NightcapGenerateCache(uint32_t *cache, uint8_t* const seed, uint64_t cache_size)
{
	uint64_t items = cache_size / NIGHTCAP_HASH_BYTES;
	sph_blake256_context ctx_blake;
	int64_t hashwords = NIGHTCAP_HASH_BYTES / NIGHTCAP_WORD_BYTES;

	sph_blake256_context ctx;
	sph_blake256_init(&ctx);
	sph_blake256(&ctx, seed, NIGHTCAP_HASH_BYTES);
	sph_blake256_close(&ctx, cache);

	for (uint64_t i = 1; i < items; i++) {
		sph_blake256_init(&ctx);
		sph_blake256(&ctx, cache + ((i - 1) * (hashwords)), NIGHTCAP_HASH_BYTES);
		sph_blake256_close(&ctx, cache + i*hashwords);
	}
	for (uint64_t round = 0; round < NIGHTCAP_CACHE_ROUNDS; round++) {
		//3 round randmemohash.
		for (uint64_t i = 0; i < items; i++) {
			uint64_t target = cache[(i * (NIGHTCAP_HASH_BYTES / sizeof(uint32_t)))] % items;
			uint64_t mapper = (i - 1 + items) % items;
			/* Map target onto mapper, hash it,
			* then replace the current cache item with the 32 byte result. */
			uint32_t item[NIGHTCAP_HASH_BYTES / sizeof(uint32_t)];
			for (uint64_t dword = 0; dword < (NIGHTCAP_HASH_BYTES / sizeof(uint32_t)); dword++) {
				item[dword] = cache[(mapper * (NIGHTCAP_HASH_BYTES / sizeof(uint32_t))) + dword]
				            ^ cache[(target * (NIGHTCAP_HASH_BYTES / sizeof(uint32_t))) + dword];
			}
			sph_blake256_init(&ctx);
			sph_blake256(&ctx, item, NIGHTCAP_HASH_BYTES);
			sph_blake256_close(&ctx, item);
			memcpy(cache + (i * (NIGHTCAP_HASH_BYTES / sizeof(uint32_t))), item, NIGHTCAP_HASH_BYTES);
		}
	}
}


/*
{
	sph_blake256_context ctx_blake;
	uint32_t const num_nodes = (uint32_t)(cache_size / sizeof(NightcapNode));
	NightcapNode *cache_nodes = (NightcapNode *)cache_nodes_in;
	sph_blake256_init(&ctx_blake);
	sph_blake256(&ctx_blake, seedhash, 32);
	sph_blake256_close(&ctx_blake, cache_nodes[0].bytes);

	for(uint32_t i = 1; i != num_nodes; ++i)
	{
		sph_blake256_init(&ctx_blake);
		sph_blake256(&ctx_blake, cache_nodes[i - 1].bytes, 32);
		sph_blake256_close(&ctx_blake, cache_nodes[i].bytes);
	}

	for(uint32_t j = 0; j < 3; j++) // this one can be unrolled entirely, ETHASH_CACHE_ROUNDS is constant
	{
		for(uint32_t i = 0; i != num_nodes; i++)
		{
			uint32_t const idx = cache_nodes[i].words[0] % num_nodes;
			NightcapNode data;
			data = cache_nodes[(num_nodes - 1 + i) % num_nodes];
			for(uint32_t w = 0; w != 16; ++w) // this one can be unrolled entirely as well
			{
				data.words[w] ^= cache_nodes[idx].words[w];
			}
			sph_blake256_init(&ctx_blake);
			sph_blake256(&ctx_blake, data.bytes, 32);
			sph_blake256_close(&ctx_blake, cache_nodes[i].bytes);
		}
	}
}
*/

