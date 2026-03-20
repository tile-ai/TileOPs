#include <tl_templates/cuda/instruction/wgmma.h>
#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void _conv2d_1x1_main_kernel(const half_t* __restrict__ bias, __grid_constant__ const CUtensorMap out_desc, __grid_constant__ const CUtensorMap weight_desc, __grid_constant__ const CUtensorMap x_desc);
extern "C" __global__ void __launch_bounds__(256, 1) _conv2d_1x1_main_kernel(const half_t* __restrict__ bias, __grid_constant__ const CUtensorMap out_desc, __grid_constant__ const CUtensorMap weight_desc, __grid_constant__ const CUtensorMap x_desc) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float out_local[32];
  tl::GmmaDescriptor desc_a;
  tl::GmmaDescriptor desc_b;
  half_t out_shared_local_cast[2];
  __shared__ __align__(16) uint64_t mbarrier_mem[4];
  auto mbarrier = reinterpret_cast<Barrier*>(mbarrier_mem);
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(weight_desc);
    tl::prefetch_tma_descriptor(x_desc);
    tl::prefetch_tma_descriptor(out_desc);
    mbarrier[0].init(128);
    mbarrier[1].init(1);
    mbarrier[2].init(128);
    mbarrier[3].init(128);
  }
  tl::fence_barrier_init();
  __syncthreads();
  if (128 <= ((int)threadIdx.x)) {
    tl::warpgroup_reg_dealloc<24>();
    const dim3 blockIdx = tl::rasterization2DRow<10>();
    mbarrier[2].wait(1);
    if (tl::tl_shuffle_elect<128>()) {
      mbarrier[0].expect_transaction(8192);
      tl::fence_proxy_async();
      tl::tma_load(weight_desc, mbarrier[0], (&(((half_t*)buf_dyn_shmem)[0])), 0, (((int)blockIdx.y) * 64));
      mbarrier[0].expect_transaction(8192);
      tl::fence_proxy_async();
      tl::tma_load(x_desc, mbarrier[0], (&(((half_t*)buf_dyn_shmem)[8192])), (((int)blockIdx.x) * 64), 0, ((int)blockIdx.z));
    }
    mbarrier[0].arrive();
  } else {
    tl::warpgroup_reg_alloc<240>();
    const dim3 blockIdx = tl::rasterization2DRow<10>();
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      float broadcast_var = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(out_local + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
    }
    mbarrier[0].wait(0);
    tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a, (&(((half_t*)buf_dyn_shmem)[0])));
    tl::initialize_wgmma_descriptor<1, 0, 64>(desc_b, (&(((half_t*)buf_dyn_shmem)[8192])));
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(out_local + 0), 32);
    tl::warpgroup_arrive();
    tl::fence_proxy_async();
    #pragma unroll
    for (int ki = 0; ki < 4; ++ki) {
      tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(uint64_t(desc_a + ((ki * 32) >> 4)), uint64_t(desc_b + ((ki * 2048) >> 4)), ((uint32_t*)(out_local + 0)), 1);
    }
    tl::warpgroup_commit_batch();
    tl::warpgroup_wait<0>();
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(out_local + 0), 32);
    mbarrier[2].arrive();
    tl::__sync_thread_partial<3, 128>();
    #pragma unroll
    for (int i_1 = 0; i_1 < 16; ++i_1) {
      uint1 __1;
      float2 __2;
        float2 v_ = *(float2*)(out_local + (i_1 * 2));
        float2 v__1 = make_float2(((float)bias[((((((int)blockIdx.y) * 64) + ((((int)threadIdx.x) >> 5) * 16)) + ((i_1 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]), ((float)bias[((((((int)blockIdx.y) * 64) + ((((int)threadIdx.x) >> 5) * 16)) + ((i_1 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]));
        __2.x = (v_.x+v__1.x);
        __2.y = (v_.y+v__1.y);
      ((half2*)(&__1))[0] = __float22half2_rn(((float2*)(&__2))[0]);
      *(uint1*)(out_shared_local_cast + 0) = __1;
      *(uint1*)(((half_t*)buf_dyn_shmem) + ((((((((((int)threadIdx.x) >> 5) * 1024) + ((i_1 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_1 >> 3)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_1 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_1 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(out_shared_local_cast + 0);
    }
    tl::fence_proxy_async();
    tl::__sync_thread_partial<3, 128>();
    if (tl::tl_shuffle_elect<128>()) {
      tl::tma_store(out_desc, (&(((half_t*)buf_dyn_shmem)[0])), (((int)blockIdx.x) * 64), (((int)blockIdx.y) * 64), ((int)blockIdx.z));
      tl::tma_store_arrive();
      tl::tma_store_wait<0>();
    }
  }
}

