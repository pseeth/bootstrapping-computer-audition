#!/bin/sh
shopt -s globstar nocaseglob
dest=music_separated_wav
num_workers=20

tsp -S $num_workers
for input in /home/data/music_separated/**/*.flac
do
  indir=$(dirname "$input")
  outdir=${indir/music_separated/$dest}
  [ ! -d "$outdir" ] && mkdir -p "$outdir"
  infile=$(basename "$input")
  outfile=${infile%.???}.wav
  tsp ffmpeg -i "$input" -ar 16000 -ac 1 "${outdir}/${outfile}"
done
# set it back to something reasonable
tsp -S 5