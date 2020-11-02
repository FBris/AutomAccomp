so=`uname -s`

if test "x$so" = xDarwin; then
  glibtoolize --copy
else
  libtoolize --copy
fi

aclocal
autoconf
autoheader
automake --copy --add-missing
