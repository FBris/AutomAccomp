AM_CPPFLAGS = -I$(top_builddir)/src/common \
	-I$(top_builddir)/src/GPU \
	$(alsa_inc_flags) \
	$(liblo_inc_flags) \
	$(use_cuda_flag) \
	$(CUDA_INCLUDE)
                        
AM_LDFLAGS = $(alsa_lib_flags) $(liblo_lib_flags)
                        
bin_PROGRAMS = ServerGPU
ServerGPU_SOURCES = ServerGPU.cu GPUFunctions.cu

.cu.o:
	$(NVCC) -o $@ -c $<  \
		$(CUDA_CFLAGS) $(AM_CPPFLAGS) $(CUDA_LIBS) $(AM_LDFLAGS) 

ServerGPU$(EXEEXT): ServerGPU.o GPUFunctions.o
	$(NVCC) -o ServerGPU ServerGPU.o GPUFunctions.o ../common/FileFunctions.c  \
		 $(CUDA_CFLAGS) $(AM_CPPFLAGS) $(CUDA_LIBS) $(AM_LDFLAGS)

