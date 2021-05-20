clear;
clc;

coords = [];
z=1;

for i=0:0.05:1
    for j=0:0.05:1
        coords(z,:) = [i j];
        z = z + 1;
    end
end
coords;
figure(1);
trimesh(delaunay(coords),coords(:,1),coords(:,2))
figure(2);
sinecoords = coords;
sinecoords(:,1) = sinecoords(:,1) + 0.10*cos(5*sinecoords(:,2));
sinecoords(:,2) = sinecoords(:,2) + 0.05*cos(7*sinecoords(:,1));
trimesh(delaunay(sinecoords),sinecoords(:,1),sinecoords(:,2))

