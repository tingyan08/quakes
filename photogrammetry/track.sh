#!/bin/zsh

set -e

params=(--dist_feature=4 --feature_radius=5 --scale=8 --search_radius=200 -S 20 -M 40000 -n 1 --corr_threshold=.85 --only-good-features -w 8 -h 1)

FEATURE_TRACK=(rh7 ~/src_boats/img_any/build/binsrc/feature_track)

function imagecode() {
    printf "%05d" $1
}

function imagefilename() {
    printf "G%07d.JPG" $1
}

function postfilter() {
    awk '/#ID/,0' | vnl-filter -p 'ID,Feat[12]'
}

ifirst=20055
ilast=20477

ihere=$ifirst
ihere_code=`imagecode $ihere`

$FEATURE_TRACK $params `imagefilename $ihere` `imagefilename $(($ihere+1))` |
    postfilter > ${ihere_code}.vnl

for ((ihere=$(($ifirst+1)); ihere<$ilast; ihere++)) {
        echo "starting $ihere"
        ihere_code=`imagecode $ihere`
        inext_code=`imagecode $(($ihere+1))`
        iprev_code=`imagecode $(($ihere-1))`
        $FEATURE_TRACK $params `imagefilename $ihere` `imagefilename $(($ihere+1))` \
                       <( < ${iprev_code}.vnl vnl-filter -p ID,Feat2) |
            postfilter > ${ihere_code}-continued.vnl

        $FEATURE_TRACK $params `imagefilename $ihere` `imagefilename $(($ihere+1))` |
            postfilter > ${ihere_code}-independent.vnl

        python get_all_new_points.py --threshold 40 ${ihere_code}-independent.vnl ${ihere_code}-continued.vnl > ${ihere_code}-new.vnl
        lastID=`tail -n1 ${iprev_code}.vnl | awk '{print $1}'`
        echo "lastID: $lastID"
        cat ${ihere_code}-continued.vnl > ${ihere_code}.vnl
        < ${ihere_code}-new.vnl vnl-filter -p "ID=NR+$lastID,Feat" >> ${ihere_code}.vnl
    }
