clear;
clc;
close all;

% Constants
MAX_ITER=100    % Max iteration count
xsize=5         % Num particles x
ysize=5         % Num particles y
n=xsize*ysize;  % Total num particles

% Swarm parameters
alpha=0.9   % Learning rate
omega=1     % Current direction weight
phi=1       % Local best direction weight
psi=1       % Global best direction weight
gamma=1     % Random direction weight

% Aggregate into weights vector
weights=[alpha omega phi psi,gamma];
% Normalize omega, phi, psi
weights(2:5)=weights(2:5)/sum(weights(2:5));



% Define arbitrary convex function
f = @(x,y) 3*(1-x).^2.*exp(-(x.^2) - (y+1).^2) - 10*(x/5 - x.^3 - y.^5).*exp(-x.^2-y.^2) - 1/3*exp(-(x+1).^2 - y.^2);
%f = @(x,y) sinc(3.*x) + sinc(3.*y);
%f = @(x,y) -((x.^(4))/(12)+3.*x.^(2).*y.^(4)-(sinc(5.*y))-cos(5.*x))

% Define bounds
xmax=2;
xmin=-xmax;
xh=0.05;
ymax=2;
ymin=-ymax;
yh=0.05;
bounds = [xmin,xmax;ymin,ymax];

% Create environment
[x, y]=meshgrid([xmin:xh:xmax],[ymin:yh:ymax]);
z=f(x,y);

% Display contour of environment
figure();
contour(x,y,z,50);
title('Particle Swarm Exploring the Function Space')
xlim([xmin,xmax]);
ylim([ymin,ymax]);
hold on;

% Create particles
for i=1:(xsize*ysize)
    % Spread particles uniformly on domain
    xpos= (xmin+(2*xmax/(xsize+1)))+(2*xmax/(xsize+1))*(mod(i-1,xsize));
    ypos= (ymin+(2*ymax/(ysize+1)))+((2*ymax/(ysize+1))*floor((ysize)*(i-1)/n));
    xvel= -1+2*rand(1);
    yvel= -1+2*rand(1);
    % Instantiate particle
    particles(i)=struct('x',xpos,'y',ypos,'f',f(xpos,ypos), ...
        'xvel',xvel,'yvel',yvel,'xbest',xpos,'ybest',ypos,...
        'fbest',f(xpos,ypos));
end

% Define swarm
swarm=struct('xpos',[particles.x],'ypos',[particles.y],...
    'xvel',[particles.xvel],'yvel',[particles.yvel],'f',[particles.f],...
    'localxbest',[particles.xbest],'localybest',[particles.ybest],...
    'globalxbest',0,'globalybest',0,'globalfbest',0);

swarm=swarmupdate(particles);

% First Display
sc=scatter(swarm.xpos,swarm.ypos,60,'filled');
hold on;
q=quiver(swarm.xpos,swarm.ypos,swarm.xvel,swarm.yvel,0.1);
q.LineWidth=2;
q.ShowArrowHead='off';
q.Clipping='on';

% Initialize some statistics arrays
fmax=zeros([MAX_ITER,1]);
delfmax=zeros([MAX_ITER,1]);
variance=zeros([MAX_ITER,1]);

% Run the simulation
for j=2:MAX_ITER
    for i=1:n
        [swarm, particles]= ...
            particleupdate(weights,f,swarm,particles,i,bounds);
    end
    pause(0.00000001);
    fmax(j)=swarm.globalfbest;
    delfmax(j)=fmax(j)/fmax(j-1);
    variance(j)=var(swarm.f-swarm.globalfbest);
    [sc, q] = displayswarm(swarm,sc,q);
end

% Do some visualization
figure();
mesh(z);
title('3D Plot of f(x,y)')
figure();
plot(fmax,'.-','markersize',10)
title('Global Max')
xlabel('Iteration')
ylabel('max f \forall particles')
figure();
plot(variance,'.-','markersize',10)
title('Variance of distance from Global Max')
xlabel('Iteration')
figure();
plot(delfmax(2:MAX_ITER),'.-','markersize',10)
title('Global Max Change Ratio')
xlabel('Iteration')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Function defs %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Update Particle
function [swarm, particles] = particleupdate(weights,f,swarm,particles,i,bounds)
alpha=weights(1);
omega=weights(2);
phi=weights(3);
psi=weights(4);
gamma=weights(5);
% Update velocity components as a random linear combination of
% current direction, direction to particle best,
% direction to swarm best, random perturbation.
particles(i).xvel=omega*rand(1)*(particles(i).xvel) + ...
    phi*rand(1)*(particles(i).xbest-particles(i).x) + ...
    psi*rand(1)*(swarm.globalxbest-particles(i).x) + ...
    gamma*(1-2*(rand(1)));
particles(i).yvel=omega*rand(1)*(particles(i).yvel) + ...
    phi*rand(1)*(particles(i).ybest-particles(i).y) + ...
    psi*rand(1)*(swarm.globalybest-particles(i).y) + ...
    gamma*(1-2*(rand(1)));
% Update local x, y
particles(i).x=particles(i).x+alpha*particles(i).xvel;
particles(i).y=particles(i).y+alpha*particles(i).yvel;

% Update local f
particles(i).f=f(particles(i).x,particles(i).y);
% Update local best
if (particles(i).f > particles(i).fbest)
    particles(i).fbest=particles(i).f;
    % Update xbest, ybest
    particles(i).xbest=particles(i).x;
    particles(i).ybest=particles(i).y;
end
% Update swarm
swarm=swarmupdate(particles);
end


% Update Swarm
function sw = swarmupdate(p)
sw.xpos=[p.x];                      % Update x positions
sw.ypos=[p.y];                      % Update y positions
sw.xvel=[p.xvel];                   % Update x velocities
sw.yvel=[p.yvel];                   % Update y velocities
sw.f=[p.f];                         % Update f values
sw.localxbest=[p.xbest];            % Update local best x
sw.localybest=[p.ybest];            % Update local y best
[best, pbest] = max([p.fbest]);     % Calculate global f max
sw.globalxbest = p(pbest).xbest;    % Update global x(fmax)
sw.globalybest= p(pbest).ybest;     % Update global y(fmax)
sw.globalfbest=best;                % Update global f max
end

% Display swarm
function [sc, q] = displayswarm(swarm,sc,q)
% Clear current swarm from display
set(sc,'Visible','off')
set(q,'Visible','off')
sc=scatter(swarm.xpos,swarm.ypos,60,'blue','filled');
hold on;
q=quiver(swarm.xpos,swarm.ypos,swarm.xvel,swarm.yvel,0.1);
q.LineWidth=2;
q.ShowArrowHead='off';
q.Color='red';
q.Clipping='on';
end