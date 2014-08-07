#include "common.h"

extern "C" void step (
		FN_TYPE *nFn_src,
		FN_TYPE *cFn_src,
		FN_TYPE *nFn_dst,
		FN_TYPE *cFn_dst,
		uint *fv,
		FN_TYPE *fv_weights,
		uint fv_pitchInBytes,
		uint *nbr,
		FN_TYPE *vtxW,
		uint vw_pitchInBytes,
		FN_TYPE *vertex_weights,
		uint vv_pitchInBytes,
		uint vv_size,
		float4 *grads,
		uint he_pitchInBytes,
		uint *parts_n,
		uint *block_face_count,
		uint blocks,
		uint threads,
		double dt,
		uint smem_size,
		bool one
		);

extern "C" void bindN1Texture(FN_TYPE *cuArray, cudaChannelFormatDesc channelDesc, size_t size);

extern "C" void bindC1Texture(FN_TYPE *cuArray, cudaChannelFormatDesc channelDesc, size_t size);

extern "C" void bindN2Texture(FN_TYPE *cuArray, cudaChannelFormatDesc channelDesc, size_t size);

extern "C" void bindC2Texture(FN_TYPE *cuArray, cudaChannelFormatDesc channelDesc, size_t size);

extern "C" void unbindN1Texture();

extern "C" void unbindC1Texture();

extern "C" void unbindN2Texture();

extern "C" void unbindC2Texture();
