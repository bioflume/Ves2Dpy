clear; clc;
oc = curve_py;

% Create geometry for confinement
prams.Nbd = 128;
prams.nvbd = 2;
t = (0:prams.Nbd-1)'/prams.Nbd * 2 * pi;
rad1 = 1; % inner cylinder radius
rad2 = 2; % outer cylinder radius
x = [rad2*cos(t) rad1*cos(-t)];
y = [rad2*sin(t) rad1*sin(-t)];
Xwalls = [x;y];
% Note: here the directions of the normals to the walls are crucial, see
% BIEM theory


% Create vesicles
prams.N = 32;
prams.nv = 2;
X0 = oc.ellipse(prams.N,0.65);
% scale X to have unit length
[~,~,len] = oc.geomProp(X0); 
X0 = X0/len;
% position X between two cylinders on the right
X = [[X0(1:end/2) + 1.5; X0(end/2+1:end)] [X0(1:end/2); X0(end/2+1:end)+1.2]];

% plot the configuration
% figure(1); clf;
% plot([Xwalls(1:end/2,:);Xwalls(1,:)], [Xwalls(end/2+1:end,:);Xwalls(end/2+1,:)],'k','linewidth',2)
% hold on
% plot([X(1:end/2,:);X(1,:)],[X(end/2+1:end,:);X(end/2+1,:)],'r','linewidth',2)
% axis equal
% pause


% the rest of the parameters and options
prams.dt = 1E-5;
prams.T = 1E+5 * prams.dt;
prams.kappa = 1;
prams.viscCont = ones(prams.nv,1);
prams.gmresTol = 1e-10;
prams.areaLenTol = 1E-2;
prams.farFieldSpeed = 1000;

options.farField = 'couette';
options.repulsion = false;
options.correctShape = true;
options.reparameterization = true;
options.usePreco = true;
options.matFreeWalls = false;
options.confined = true;

[options, prams] = initVes2D(options,prams);




% The following could be a separate script commonly called for any flow
% inputting options, prams, X and Xwalls

% get original area and length
[~,area0,len0] = oc.geomProp(X);

% Create tstep_biem
tt = tstep_biem(X,Xwalls,options,prams);

if options.confined
tt.initialConfined();
end

% initialize unknowns
sigma = zeros(prams.N,prams.nv); eta = zeros(2*prams.Nbd, prams.nvbd); RS = zeros(3,prams.nvbd); 

time = 0; 
disp([num2str(prams.nv) ' vesicle(s) in ' options.farField ' flow, dt: ' num2str(prams.dt)])
disp(['Vesicle(s) discretized with ' num2str(prams.N) ' points'])
if options.confined
  disp(['Wall(s) discretized with ' num2str(prams.Nbd) ' points'])
end

while time < prams.T
  
  % take a step
  tStart = tic;
  [Xnew, sigma, eta, RS, iter, iflag] = tt.timeStep(X, sigma, eta, RS);
  tEnd = toc(tStart);

  if options.reparameterization
    % Redistribute points equally in arc-length
    XnewO = Xnew;
    for it = 1 : 5
      Xnew = oc.redistributeArcLength(Xnew);
    end
    X = oc.alignCenterAngle(XnewO,Xnew);
  end

  if options.correctShape
    % area-length correction
    X = oc.correctAreaAndLength(X,area0,len0);
  end

  time = time + prams.dt;
  disp('*****************************************************************')
  disp(['Time: ' num2str(time) ' out of Tf: ' num2str(prams.T)])
  disp(['GMRES took ' num2str(iter) ' iterations.'])
  disp(['Time step takes ' num2str(tEnd) ' seconds'])
  disp('*****************************************************************')

  figure(1); clf;
  plot([Xwalls(1:end/2,:);Xwalls(1,:)], [Xwalls(end/2+1:end,:);Xwalls(end/2+1,:)],'k','linewidth',2)
  hold on
  plot([X(1:end/2,:);X(1,:)],[X(end/2+1:end,:);X(end/2+1,:)],'r','linewidth',2)
  axis equal
  pause(0.1)

end




