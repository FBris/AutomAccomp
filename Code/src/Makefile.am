AM_CPPFLAGS = $(use_cpu_flag) \
	$(use_cuda_flag) 

SUBDIRS=

if With_CPU
  SUBDIRS += CPU
endif

if With_CUDA
  SUBDIRS += GPU
endif
