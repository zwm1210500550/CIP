#!/bin/bash

sed '/^$/d' data.conll > data;
awk '{print $2}' data > word.dict;

awk '{print $1,$2}' data.conll > data;
awk '{if($1>0) printf $2; else print ''}' data > data.txt;

rm data;


