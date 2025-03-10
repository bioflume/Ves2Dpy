clear; clc;

oc = curve_py;

% Create perturbed vesicle configuration:
N = 32; % number of points to discretize vesicle
t = (0:N-1)'*2*pi/N;
a = 1; b = 3*a; c = 0.85; 
r = 0.55*sqrt( (a*cos(t)).^2 + (b*sin(t)).^2) + .07*cos(12*(t));
x = c*r.*cos(t);
y = r.*sin(t);
[~, ~, len] = oc.geomProp([x;y]); 

X = 1/len * [x;y]; % Vesicle
nv = 1; % number of vesicles
[ra, area, len] = oc.geomProp(X); % Geometric properties (unit length)

% we will take some time steps to relax if we want to find shape evolution
dt = 1E-5;
Th = 1000 * dt; 
time = (0:dt:Th);
op = poten_py(N); % needed to build Stokes single layer integral matrix

%% time stepping
% this can be done matrix-free as well, however, since the matrix is of
% size 2*N x 2*N and N = 32, it might be fast to build and solve with
% matrix (?)

% I leave speeding this up to you. 

for tt = 1 : numel(time)
  vesicle = capsules_py(X,[],[],1,ones(nv,1));  
  [Ben, Ten, Div] = vesicle.computeDerivs();
  G = op.stokesSLmatrix(vesicle);
  Xnew = zeros(size(X));
  for k = 1 : nv
    M = G(:,:,k)*Ten(:,:,k)*((Div(:,:,k)*G(:,:,k)*Ten(:,:,k))\eye(vesicle.N))*Div(:,:,k);
    LHS = (eye(2*vesicle.N)-vesicle.kappa*dt*(-G(:,:,k)*Ben(:,:,k)+M*G(:,:,k)*Ben(:,:,k)));
    Xnew(:,k) = LHS\X(:,k);
  end
  X = Xnew;
  figure(1);clf;
  plot(X(1:end/2,:),X(end/2+1:end,:))
  axis equal
  pause(0.1)
end




