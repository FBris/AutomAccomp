AM_CPPFLAGS = -I$(top_builddir)/src/common \
	-I$(top_builddir)/src/CPU \
	$(alsa_inc_flags) \
	$(fftw_inc_flags) \
	$(liblo_inc_flags) \
	$(cblas_inc_flags) \
	$(use_cpu_flag) 


AM_LDFLAGS = $(alsa_lib_flags) \
	$(liblo_lib_flags) \
	$(fftw_lib_flags) \
	$(cblas_lib_flags)


bin_PROGRAMS = ServerCPU
ServerCPU_SOURCES = ServerCPU.c CPUFunctions.c ../common/FileFunctions.c
