
#draw ellipse
gmt psxy -R-2/2/-2/2 -Jx3 -Se -Wthicker,black -K > perturb.ps <<EOF
0 0 20 3.6 2.8
EOF

gmt psxy -R -J -Sv0.3+e -Wthicker,red -K -O >> perturb.ps <<EOF
0 0 20 1.8 1.8
EOF
gmt psxy -R -J -Sv0.3+e -Wthicker,blue -K -O >> perturb.ps <<EOF
0 0 110 1.4 1.4
EOF

gmt pstext -R -J -K  -O >>perturb.ps <<EOF
.0 0.36 @~l@-3
EOF

gmt pstext -R -J -K -O >>perturb.ps <<EOF
.32 0.26 @~l@-1
EOF

gmt pstext -R -J -K -O >>perturb.ps <<EOF
.46 0.10 @~q
EOF

gmt psxy -R -J -Sv0.3+e -Wthicker,black -K -O >> perturb.ps <<EOF
0 0 4 2.4 1 
EOF

gmt pstext -R -J -K -O >>perturb.ps <<EOF
0.9 .05 @~W
EOF

gmt psxy -R -J -Sc.1i -Gdarkgray -K -O >>perturb.ps <<EOF
-0.2 0.3
EOF

#draw ellipse
gmt psxy -R-2/2/-2/2 -Jx3 -X2i -Se -Wthicker,black -K -O >> perturb.ps <<EOF
0 0 70 3.2 3.2
EOF

gmt psxy -R -J -Sv0.3+e -Wthicker,red -K -O >> perturb.ps <<EOF
0 0 80 1.6 1.6
EOF
gmt psxy -R -J -Sv0.3+e -Wthicker,blue -K -O >> perturb.ps <<EOF
0 0 170 1.6 1.6
EOF

gmt pstext -R -J -K  -O >>perturb.ps <<EOF
-0.36 0.2 @~l@-3
EOF

gmt pstext -R -J -K -O >>perturb.ps <<EOF
.0 0.46 @~l@-1
EOF

gmt pstext -R -J -K -O >>perturb.ps <<EOF
.1 0.10 @~q
EOF

gmt psxy -R -J -Sv0.3+e -Wthicker,black -K -O >> perturb.ps <<EOF
0 0 4 2.4 1 
EOF

gmt pstext -R -J -K -O >>perturb.ps <<EOF
0.9 .05 @~W
EOF

gmt psxy -R -J -Sc.1i -Gdarkgray -K -O >>perturb.ps <<EOF
-0.2 0.3
EOF


gmt psxy -T -R -J -O>>perturb.ps

gmt ps2raster -Tf -A -P perturb.ps 
rm perturb.ps gmt.history
