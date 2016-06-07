
#draw ellipse
gmt psxy -R-2/2/-2/2 -Jx3 -Se -Wthicker,black -K > perturb.ps <<EOF
0 0 20 3.6 2.8
EOF

# red axis
gmt psxy -R -J -Sv0.3+e -Wthicker,red -K -O >> perturb.ps <<EOF
0 0 20 1.8 1.8
EOF
#blue axis
gmt psxy -R -J -Sv0.3+e -Wthicker,blue -K -O >> perturb.ps <<EOF
0 0 290 1.4 1.4
EOF

#lambda 1
gmt pstext -R -J -K  -O -F+f9 >>perturb.ps <<EOF
.04 -0.32 @~l@-1@-@~
EOF

#lambda 2
gmt pstext -R -J -K -O -F+f9 >>perturb.ps <<EOF
.32 0.26 @~l@-2@-@~
EOF

#theta
gmt pstext -R -J -K -O -F+f9 >>perturb.ps <<EOF
.46 0.10 @~x@~
EOF

#Rotation axis
gmt psxy -R -J -Sv0.3+e -Wthicker,black -K -O >> perturb.ps <<EOF
0 0 4 2.4 1 
EOF

#Omega
gmt pstext -R -J -K -O >>perturb.ps <<EOF
0.9 .05 @~W@~
EOF

#perturbation
gmt psxy -R -J -Sc.1i -Gdarkgray -K -O >>perturb.ps <<EOF
-0.2 0.3
EOF

#draw ellipse
gmt psxy -R-2/2/-2/2 -Jx3 -X2i -Se -Wthicker,black -K -O >> perturb.ps <<EOF
0 0 70 3.2 3.2
EOF

#red axis
gmt psxy -R -J -Sv0.3+e -Wthicker,red -K -O >> perturb.ps <<EOF
0 0 80 1.6 1.6
EOF
#blue axis
gmt psxy -R -J -Sv0.3+e -Wthicker,blue -K -O >> perturb.ps <<EOF
0 0 350 1.6 1.6
EOF

#lambda 1
gmt pstext -R -J -K  -O -F+f9 >>perturb.ps <<EOF
.36 -0.15 @~l@-1@-@~
EOF

#lambda 2
gmt pstext -R -J -K -O -F+f9 >>perturb.ps <<EOF
-.02 0.46 @~l@-2@-@~
EOF

#theta
gmt pstext -R -J -K -O -F+f9 >>perturb.ps <<EOF
.1 0.10 @~x@~
EOF

#rotation axis
gmt psxy -R -J -Sv0.3+e -Wthicker,black -K -O >> perturb.ps <<EOF
0 0 4 2.4 1 
EOF

#Omega
gmt pstext -R -J -K -O >>perturb.ps <<EOF
0.9 .05 @~W@~
EOF

#perturbation
gmt psxy -R -J -Sc.1i -Gdarkgray -K -O >>perturb.ps <<EOF
-0.2 0.3
EOF


gmt psxy -T -R -J -O>>perturb.ps

gmt ps2raster -Tf -A -P perturb.ps 
rm perturb.ps gmt.history
