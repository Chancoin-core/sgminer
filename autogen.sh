#!/bin/sh
unamestr=`uname`

if [ "$unamestr" = 'Darwin' ]; then
bs_dir="$(dirname $(greadlink -f $0))"
else
bs_dir="$(dirname $(readlink -f $0))"
fi

#Some versions of libtoolize don't like there being no ltmain.sh file already
touch "${bs_dir}"/ltmain.sh
autoreconf -fi "${bs_dir}"
