#!/bin/bash

usage() { echo "Usage: $0 [-c CONF_FILE] [-r REPEAT] [-h]
    -c  configuration file
    -r  repeat script execution this number of times
    -h  prints this help message
" 1>&2;}

conf=lamina_default.cfg
rep=1

while getopts "hc:r:" arg; do
    case "$arg" in
        c)
            conf=${OPTARG}
            ;;
        r)
            rep=${OPTARG}
            ;;
        h)
            usage
            exit
            ;;
        *)
            usage
            exit 1
            ;;
    esac
done

# add extension if not exists
if [[ $conf != *.cfg ]]; then
    file=$conf.cfg
fi

# create default(empty) configuration if file not exists
if [ ! -f $conf ]; then
    echo 'Configuration file' $conf 'does not exist. Generating ' \
        'default configuration.'
    touch $conf
fi


echo 'Using' $conf 'configuration file, executing' $rep 'times'
if (( $rep > 1 )) ; then
  for (( i=0; i<$rep; i++ ))
  do
    echo 'Execution:' $((i + 1))
    python lamina_demo.py -c $conf -v $i
  done
else
  python lamin_demo.py -c $conf -v -1
fi

