classdef tstep_biem < handle
% This class defines the functions required to advance the geometry
% forward in time.  Handles both implicit and explicit vesicle-vesicle
% interactions, different inextensibility conditions, viscosity
% contrast, solid walls vs. unbounded flows.  This class also
% implements the adaptive time stepping strategy where the errors in
% length, area, and the residual are monitored to adaptively choose a
% time step size.


properties
Xwalls        % Points coordinates on walls
walls         % walls capsules class
area          % vesicles' initial area
length        % vesicles' initial length
dt            % Time step size
currentTime   % current time needed for adaptive time stepping
finalTime     % time horizon

Galpert       % Single-layer stokes potential matrix using Alpert
D             % Double-layer stokes potential matrix
lapDLP        % Double-layer laplace potential matrix
DLPnoCorr     % Double-layer stokes potential matrix without correction
SLPnoCorr     % Single-layer stokes potential matrix without correction

wallDLP       % Double-layer potential due to solid walls
wallN0        % Modificiation of double-layer potential to 
              % remove the rank deficiency on outer boundary
                  
farField      % Far field boundary condition
confined      % whether geometry is bounded or not


bdiagVes      % precomputed inverse of block-diagonal precondtioner
              % only for vesicle-vesicle interactions
bdiagTen
bdiagWall     % precomputed inverse of block-diagonal precondtioner
              % only for wall-wall interactions
              

gmresTol      % GMRES tolerance
gmresMaxIter  % maximum number of gmres iterations

tstepTol      % maximum allowable error in area and length

NearV2V       % near-singular integration for vesicle to vesicle
NearW2V       % near-singular integration for wall to vesicle
NearV2W       % near-singular integration for vesicle to wall 
NearW2W       % near-singular integration for wall to wall 

op            % class for evaluating potential so that we don't have
              % to keep building quadrature matricies
opWall        % class for walls              


repulsion     % use repulsion in the model
repStrength   % repulsion strength
minDist       % minimum distance to activate repulsion

matFreeWalls  % Compute wall-wall interactions matrix free
wallDLPandRSmat % wall2wall interaction matrix computed in initialConfined


haveWallMats  % do we have wall matrices computed before?

usePreco     % use block-diagonal preco?


% These might be saved when we are scaling RHS for Wall2Wall interactions
% and hence do not compute again, just scale it
eta
RS

% Inverse of the blocks of wall-2-wall interaction matrix
invM11
invM22

end % properties

methods

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function o = tstep_biem(X,Xwalls,options,prams)
oc = curve_py;

o.Xwalls = Xwalls; % points on walls
[~,o.area,o.length] = oc.geomProp(X);

o.dt = prams.dt; % Time step size

% Method always starts at time 0
o.currentTime = 0;

% Need the time horizon for adaptive time stepping
o.finalTime = prams.T;


% GMRES tolerance
o.gmresTol = prams.gmresTol;
% maximum number of GMRES iterations
o.gmresMaxIter = prams.gmresMaxIter;

% Far field boundary condition built as a function handle
o.farField = @(X,Xwalls) o.bgFlow(X,Xwalls,options.farField,...
    'Speed',prams.farFieldSpeed,'chanWidth',prams.chanWidth,'vortexSize',...
    prams.vortexSize);

% Confined or unbounded geometry
o.confined = ~isempty(Xwalls);

% Time step tolerance  
o.tstepTol = prams.tstepTol;

% Repulsion between vesicles and vesicles-walls.
% if there is only one vesicle, turn off repulsion
o.repulsion = options.repulsion;
if prams.nv == 1 && ~o.confined
  o.repulsion = false;
end
% scaling the strength of the repulsion
o.repStrength = prams.repStrength;
% scaling the range of repulsion
o.minDist = prams.minDist;

% build poten class for vesicles
o.op = poten_py(prams.N);

% use preconditioner?
o.usePreco = options.usePreco;
o.bdiagVes = [];

% build poten classes for walls
if o.confined
  
  o.initialConfined(); % create wall related matrices (preconditioner, DL potential, etc)
  
  o.eta = []; % density on wall
  o.RS = []; % rotlets and stokeslets defined at the center
  
  % flag for computing the W2W interactions with a precomp. matrix or not
  % if matFreeWalls = true, then we compute W2W interactions at every time
  % step either using FMM or not. If matFreeWalls = false, then we use the
  % precomputed matrix and apply it to a vector of density and RS to get
  % the wall2wall interactions.
  o.matFreeWalls = options.matFreeWalls;  
    
else
  o.opWall = [];
end


end % tstep_biem: constructor

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function initialConfined(o)
% initialConfined(o) builds wall-related matrices

Nbd = numel(o.Xwalls(:,1))/2; % number of points per wall
nvbd = numel(o.Xwalls(1,:)); % number of walls
% if the walls are discretized with the same Nbd
o.opWall = poten_py(Nbd);
  
% velocity on solid walls coming from no-slip boundary condition
[uwalls,~] = o.farField([],o.Xwalls);

% build the walls
o.walls = capsules_py(o.Xwalls,[],uwalls,zeros(nvbd,1),zeros(nvbd,1));
    
% build the double-layer potential matrix for walls and save on memory
o.wallDLP = o.opWall.stokesDLmatrix(o.walls);
  
% N0 to remove rank-1 deficiency
o.wallN0 = potWall.stokesN0matrix(o.walls);
  
% block diagonal preconditioner for solid walls
o.bdiagWall = o.wallsPrecond();

end % initialConfined

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [X,sigma,eta,RS,iter,iflag] = timeStep(o,...
    Xstore,sigStore,etaStore,RSstore,viscCont,vesicle)

% [X,sigma,eta,RS,iter,iflag] = timeStep(o,...
%     Xstore,sigStore,etaStore,RSstore,viscCont,vesicle)
% uses implicit vesicle-vesicle interactions.  

N = size(Xstore,1)/2; % Number of points per vesicle
nv = size(Xstore,2); % Number of vesicles

Nbd = size(o.Xwalls,1)/2; % Number of points on the solid walls
% number of solid wall components
nvbd = size(o.Xwalls,2);    % # of walls of the same discretization

% constant that appears in front of time derivative in
% vesicle dynamical equations
alpha = (1 + viscCont)/2;  


% Build single layer potential matrix and put it in current object
op = o.op;
o.Galpert = op.stokesSLmatrix(vesicle); 


% Compute double-layer potential matrix due to each vesicle
% independent of the others.  Matrix is zero if there is no
% viscosity contrast
o.D = [];
if any(viscCont ~= 1)
  o.D = op.stokesDLmatrix(vesicle);
end
 

% Structures for deciding who
% is close, how close it is, who is closest, etc., needed in nearSingInt
if o.confined
    
  % Need vesicle to vesicle and vesicle to wall interactions
  [o.NearV2V,o.NearV2W] = vesicle.getZone(o.walls,3);
      
  % Only need wall to vesicle interactions.  Wall to wall
  % interactions should also use near-singular integration since
  % they may be close to one another
  if nvbd == 1
    [o.NearW2W,o.NearW2V] = o.walls.getZone(vesicle,2);
  else
    if isempty(o.NearW2W)
      [o.NearW2W,o.NearW2V] = o.walls.getZone(vesicle,3);
    else
    % there is no need to compute W2W again, since they do not move  
      [~,o.NearW2V] = o.walls.getZone(vesicle,2);
    end
  end
    
else
  % no solid walls, so only need vesicle-vesicle intearactions
  o.NearV2V = vesicle.getZone([],1);
  o.NearV2W = [];
  o.NearW2V = [];
  o.NearW2W = [];  
end
  


% Parts of rhs from previous solution.  
rhs1 = Xstore;
rhs2 = zeros(N,nv);
if o.confined
  rhs3 = o.walls.u;
else
  rhs3 = [];
end


% START TO COMPUTE RIGHT-HAND SIDE DUE TO VESICLE TRACTION JUMP
% vesicle-vesicle and vesicle-wall interactions are handled
% implicitly in TimeMatVec
% END TO COMPUTE RIGHT-HAND SIDE DUE TO VESICLE TRACTION JUMP

% START TO COMPUTE RIGHT-HAND SIDE DUE TO VISCOSITY CONTRAST
if any(vesicle.viscCont ~= 1)
  
  % Need to add jump to double-layer potential if using near-singular
  % integration so that we can compute boundary values for the
  % near-singular integration algorithm
  jump = 1/2*(1-vesicle.viscCont);
  DLP = @(X) X*diag(jump) + op.exactStokesDLdiag(vesicle,o.D,X);

  % Use near-singular integration to compute double-layer
  % potential from previous solution 
  Fdlp = op.nearSingInt(vesicle,Xstore,DLP,...
      o.NearV2V,@op.exactStokesDL,vesicle,true);
  FDLPwall = [];    
  if o.confined
    FDLPwall = op.nearSingInt(vesicle,Xstore,DLP,...
        o.NearV2W,@op.exactStokesDL,o.walls,false);
  end
    
else
  % If no viscosity contrast, there is no velocity induced due to a
  % viscosity contrast  
  Fdlp = zeros(2*N,nv);
  FDLPwall = zeros(2*Nbd,nvbd);
end

% add in viscosity contrast term due to each vesicle independent of the
% others (o.D * Xo) from the previous solution followed by the term due
% to all other vesicles (Fdlp)
if (any(viscCont ~= 1))
  DXo = op.exactStokesDLdiag(vesicle,o.D,Xstore);
  rhs1 = rhs1 - (Fdlp + DXo) * diag(1./alpha);
end

% compute the double-layer potential due to all other vesicles from the
% appropriate linear combination of previous time steps.  Depends on
% time stepping order and vesicle-vesicle discretization
rhs3 = rhs3 + FDLPwall/o.dt;

% START COMPUTING SINGLE-LAYER POTENTIAL FOR REPULSION
if o.repulsion 
  % Repulsion is handled explicitly between vesicles, vesicles-walls.  
  
  if ~o.confined
    repulsion = vesicle.repulsionScheme(Xrep,o.repStrength,o.minDist,...
        [],[],[]);
  else
    repulsion = vesicle.repulsionScheme(Xrep,o.repStrength,o.minDist,...
          o.walls,[],[]);
  end

  Frepulsion = op.exactStokesSLdiag(vesicle,o.Galpert,repulsion);
  % diagonal term of repulsion


  SLP = @(X) op.exactStokesSLdiag(vesicle,o.Galpert,X);
    
  % Use near-singular integration to compute single-layer potential
  % due to all other vesicles.  Need to pass function
  % op.exactStokesSL so that the layer potential can be computed at
  % far points and Lagrange interpolation points
  Frepulsion = Frepulsion + op.nearSingInt(vesicle,repulsion,SLP,...
        o.NearV2V,@op.exactStokesSL,vesicle,true);
    
  % Evaluate the velocity on the walls due to the vesicles
  FREPwall = [];
  if o.confined
    FREPwall = op.nearSingInt(vesicle,repulsion,SLP,...
        o.NearV2W,@op.exactStokesSL,o.walls,false);
  end
    
  rhs1 = rhs1 + o.dt*Frepulsion*diag(1./alpha);
  rhs3 = rhs3 - FREPwall;
end
% END COMPUTING SINGLE-LAYER POTENTIALS FOR REPULSION 



% START TO COMPUTE RIGHT-HAND SIDE DUE TO SOLID WALLS
% This is done implicitly in TimeMatVec
if ~o.confined
    % Add in far-field condition (extensional, shear, etc.)
  vInf = o.farField(Xstore,[]);
  rhs1 = rhs1 + o.dt*vInf*diag(1./alpha);
end
% END TO COMPUTE RIGHT-HAND SIDE DUE TO SOLID WALLS


% START TO COMPUTE THE RIGHT-HAND SIDE FOR THE INEXTENSIBILITY CONDITION
% rhs2 is the right-hand side for the inextensibility condition
rhs2 = rhs2 + vesicle.surfaceDiv(Xo); 
% END TO COMPUTE THE RIGHT-HAND SIDE FOR THE INEXTENSIBILITY CONDITION

% The next makes sure that the rhs is all order one rather than have rhs3
% being order 1/o.dt and other two parts (rhs1 and rhs2) being order 1.
% This of course needs to be compensated in the TimeMatVec routine
if (any(vesicle.viscCont ~= 1) && o.confined)
  rhs3 = rhs3 * o.dt;
end

% Stack the right-hand sides in an alternating with respect to the
% vesicle fashion
rhs = [rhs1; rhs2];
rhs = [rhs(:); rhs3(:)];

% Add on the no-slip boundary conditions on the solid walls
% Rotlet and Stokeslet equations
rhs = [rhs; zeros(3*(nvbd-1),1)];


    
% START BUILDING BLOCK-DIAGONAL PRECONDITIONER
if o.usePreco 
  
  % Build differential operators. 
  % Compute bending, tension, and surface divergence of current
  % vesicle configuration
  [Ben,Ten,Div] = vesicle.computeDerivs;

  bdiagVes.L = zeros(3*N,3*N,nv);
  bdiagVes.U = zeros(3*N,3*N,nv);
  
  % Build block-diagonal preconditioner of self-vesicle 
  % intearctions in matrix form
  for k=1:nv
    if any(vesicle.viscCont ~= 1)
      [bdiagVes.L(:,:,k),bdiagVes.U(:,:,k)] = lu(...
        [ (eye(2*N) - o.D(:,:,k)/alpha(k)) + ...
          o.dt/alpha(k)*vesicle.kappa*o.Galpert(:,:,k)*Ben(:,:,k) ...
         -o.dt/alpha(k)*o.Galpert(:,:,k)*Ten(:,:,k); ...
          Div(:,:,k) zeros(N)]);
    else
      [bdiagVes.L(:,:,k),bdiagVes.U(:,:,k)] = lu(...
        [ eye(2*N) + ...
          o.dt*vesicle.kappa*o.Galpert(:,:,k)*Ben(:,:,k) ...
         -o.dt/alpha(k)*o.Galpert(:,:,k)*Ten(:,:,k); ...
          Div(:,:,k) zeros(N)]);
    end
  end
  o.bdiagVes = bdiagVes;
end % usePreco

% SOLVING THE SYSTEM USING GMRES
% any warning is printed to the terminal and the log file so
% don't need the native matlab version
initGMRES = [Xstore;sigStore];
initGMRES = initGMRES(:);
if o.confined 
  RS = RSstore(:,2:end);
  initGMRES = [initGMRES;etaStore(:);RS(:)];
end

% Use GMRES to solve for new positions, tension, density
% function defined on the solid walls, and rotlets/stokeslets
if o.usePreco 
  [Xn,iflag,~,I,~] = gmres(@(X) o.TimeMatVec(X,vesicle),...
      rhs,[],o.gmresTol,o.gmresMaxIter,...
      @o.preconditionerBD,[],initGMRES);
  iter = I(2);    
else
  [Xn,iflag,~,I,~] = gmres(@(X) o.TimeMatVec(X,vesicle),...
      rhs,[],o.gmresTol,o.gmresMaxIter);
  iter = I(2);
end

disp(['DONE, it took ' num2str(toc(tGMRES),'%2.2e') ' seconds']);
% END OF SOLVING THE SYSTEM USING GMRES

% allocate space for positions, tension, and density function
X = zeros(2*N,nv);
sigma = zeros(N,nv);
eta = zeros(2*Nbd,nvbdSme);
RS = zeros(3,nvbd);

% unstack the positions and tensions
for k=1:nv
  X(:,k) = Xn((3*k-3)*N+1:(3*k-1)*N);
  sigma(:,k) = Xn((3*k-1)*N+1:3*k*N);
end

% unstack the density function
Xn = Xn(3*nv*N+1:end);
for k = 1:nvbd
  eta(:,k) = Xn((k-1)*2*Nbd+1:2*k*Nbd);
end


% unstack the rotlets and stokeslets
otlets = Xn(2*nvbd*Nbd+1:end);
for k = 2:nvbd
  istart = (k-2)*3+1;
  iend = 3*(k-1);
  RS(:,k) = otlets(istart:iend);
end

end % timeStep

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function val = TimeMatVec(o,Xn,vesicle)
% val = TimeMatVec(o,Xn,vesicle,walls,wallsInt,wallsExt) 
% MATVEC for GMRES in the IMEX scheme.
% Evaluations vesicle-vesicle and vesicle-boundary interaction formulas
% to the function Xn which contains both the position, tension, and
% density
% 
% - Xn : all state variables in the following order
%   ves1:x,y,sigma,  ves2:x,y,sigma, ... vesNv:x,y,sigma, (outer_solidwall1:fx,fy,
%   inner_solidwall_1:fx,fy; inner_solid_wall2:fx,fy; ...; inner_solid_walln:fx,fy;
%   stokeslet_rotlet_innerwall1, stokeslet_rolet_innerwall2....
%
% - vesicle: class capsules used to evaluate the operators for the GMRES 


% counter for the number of matrix-vector multiplications
% that are required for the entire simulation
global matvecs  
matvecs = matvecs + 1;

walls = o.walls;
op = o.op; % poten class
N = vesicle.N; % Number of points
nv = vesicle.nv; % Number of vesicles
Nbd = 0; 
nvbd = 0;
if o.confined
  Nbd = walls.N; % Number of points on walls
  nvbd = walls.nv; % Number of components to walls
end


% right-hand side that corresponds to position equation
valPos = zeros(2*N,nv);
% right-hand side that corresponds to inextensibilty equation
valTen = zeros(N,nv);
% right-hand side that corresponds to solid wall equation
if o.confined 
  valWalls = zeros(2*Nbd,nvbdSme);
  % right-hand side corresponding to the rotlets and stokeslets
  valLets = zeros(3*(nvbd-1),1);
end

% Unstack the position and tension from the input
Xm = zeros(2*N,nv);
sigmaM = zeros(N,nv);
for k=1:nv
  Xm(1:2*N,k) = Xn((3*k-3)*N+1:(3*k-1)*N);
  sigmaM(:,k) = Xn((3*k-1)*N+1:3*k*N);
end

% Unstack the density function from the input
if o.confined
    
  eta = Xn(3*nv*N+1:end);
  % x for the system DLPRSmat*x = b (W2W handled by the pre-computed matrix)
  % much faster than FMM for walls since geometries do not change. It
  % requires memory for walls with large Nbds though.
  etaAll = eta; % stacked eta  
  
  etaM = zeros(2*Nbd,nvbd);
  for k = 1:nvbd
    etaM(:,k) = eta((k-1)*2*Nbd+1:2*k*Nbd);
  end
  otlets = Xn(3*nv*N+2*nvbd*Nbd+1:end);
else
  etaM = [];
  otlets = [];
end

% otlets keeps stokeslets and rotlets of each wall. Ordered as
% [stokeslet1(component 1);stokeslet1(component 2);rotlet1;...
%  stokeslet2(component 1);stokeslet2(component 2);rotlet2;...];

% f is the traction jump stored as a 2N x nv matrix
f = vesicle.tracJump(Xm,sigmaM);

% constant that multiplies the time derivative in the 
% vesicle position equation
alpha = (1+vesicle.viscCont)/2; 

% Gf is the single-layer potential applied to the traction jump. 
Gf = op.exactStokesSLdiag(vesicle,o.Galpert,f);

% DXm is the double-layer potential applied to the position
if any(vesicle.viscCont ~= 1)
  DXm = op.exactStokesDLdiag(vesicle,o.D,Xm);
else
  DXm = zeros(2*N,nv);
end


% START COMPUTING REQUIRED SINGLE-LAYER POTENTIALS
% Evaluate single-layer potential due to all vesicles except itself and
% the single-layer potential due to all vesicles evaluated on the solid
% walls.  

% Evaulate single-layer potential due to all other vesicles
% WITH near-singular integration.  FMM is optional
SLP = @(X) op.exactStokesSLdiag(vesicle,o.Galpert,X);
Fslp = op.nearSingInt(vesicle,f,SLP,...
    o.NearV2V,@op.exactStokesSL,vesicle,true);

FSLPwall = [];
if o.confined
  % Evaluate single-layer potential due to all vesicles on
  % the solid walls WITH near-singular integration
  FSLPwall = op.nearSingInt(vesicle,f,SLP,...
    o.NearV2W,@op.exactStokesSL,walls,false);
end
% END COMPUTING REQUIRED SINGLE-LAYER POTENTIALS


% START COMPUTING REQUIRED DOUBLE-LAYER POTENTIALS FOR VISCOSITY
% CONTRAST
if any(vesicle.viscCont ~= 1)
    
  jump = 1/2*(1-vesicle.viscCont);
  DLP = @(X) X*diag(jump) + op.exactStokesDLdiag(vesicle,o.D,X);
  
  % Use near-singular integration to compute double-layer
  % potential due to V2V interactions. FMM optional.
  Fdlp = op.nearSingInt(vesicle,Xm,DLP,...
    o.NearV2V,@op.exactStokesDL,vesicle,true);
  
  FDLPwall = [];
  if o.confined
      FDLPwall = op.nearSingInt(vesicle,Xm,DLP,[],...
        o.NearV2W,@op.exactStokesDL,walls,false);
  end
  
else
  Fdlp = [];
  FDLPwall = [];
end
% END COMPUTING REQUIRED DOUBLE-LAYER POTENTIALS FOR VISCOSITY CONTRAST

% START OF EVALUATING DOUBLE-LAYER POTENTIALS DUE TO SOLID WALLS ON
% VESICLES
Fwall2Ves = zeros(2*N,nv);
if o.confined
  jump = -1/2;  
  
  potWall = o.opWall;
    
  DLP = @(X) jump*X + potWall.exactStokesDLdiag(walls,o.wallDLP,X);
  Fwall2Ves = potWall.nearSingInt(walls,etaM,DLP,...
        o.NearW2V,@potWall.exactStokesDL,vesicle,false);  
end
% END OF EVALUATING DOUBLE-LAYER POTENTIALS DUE TO SOLID WALLS

% START OF EVALUATING WALL TO WALL INTERACTIONS
if o.confined
  
  % only need to do wall to wall interactions if the domain is multiply
  % connected
  if  nvbd > 1
    if o.matFreeWalls
      potWall = o.opWall;
      FDLPwall2wall = potWall.exactStokesDL(walls,etaM,[]);
    else %o.matFreeWalls
      % !!in-core without fast-direct solver 
      % (out-core and with FD will be implemented) 
      wallAllRHS = o.wallDLPandRSmat*etaAll;  
      FDLPwall2wall = wallAllRHS(1:2*Nbd*nvbd);
      valLets = wallAllRHS(2*Nbd*nvbd+1:end);
    end % o.matFreeWalls
    
  elseif nvbd == 1
    if ~o.matFreeWalls
      wallAllRHS = o.wallDLPandRSmat*etaAll;  
      valWalls = wallAllRHS(1:2*Nbd*nvbd);
    end
  end % ~o.diffDiscWalls && nvbd>1 
end % o.confined
% END OF EVALUATING WALL TO WALL INTERACTIONS


% START OF EVALUATING POTENTIAL DUE TO STOKESLETS AND ROTLETS
if nvbd > 1

  LetsWalls = zeros(2*Nbd,nvbd);
  LetsVes = zeros(2*N,nv);
  for k = 2:nvbd
    stokeslet = otlets(3*(k-2)+1:3*(k-2)+2);
    rotlet = otlets(3*(k-1));
    % compute velocity due to rotlets and stokeslets on the vesicles
    LetsVes = LetsVes + o.RSlets(vesicle.X,walls.center(:,k),...
        stokeslet,rotlet);
    % compute velocity due to rotlets and stokeslets on the solid walls
    if o.matFreeWalls    
      LetsWalls = LetsWalls + o.RSlets(walls.X,walls.center(:,k),...
          stokeslet,rotlet);
    end
    % if ~matFreeWalls, these are already computed above
  end
  % Integral constraints on the density function eta related
  % to the weights of the stokeslets and rotlets
  if o.matFreeWalls
    valLets = o.letsIntegrals(otlets,etaM);
  end
  % if ~matFreeWalls, these are already computed above
else
  LetsVes = [];
  LetsWalls = [];
  FDLPwall2wall = [];
end
% END OF EVALUATING POTENTIAL DUE TO STOKESLETS AND ROTLETS

% START OF EVALUATING VELOCITY ON VESICLES

if ~isempty(Gf)
  % self-bending and self-tension terms
  valPos = valPos - o.dt*Gf*diag(1./alpha);
end

if ~isempty(DXm)
  % self-viscosity contrast term
  valPos = valPos - DXm*diag(1./alpha);
end
if ~isempty(Fslp)
  % single-layer potential due to all other vesicles
  valPos = valPos - o.dt*Fslp*diag(1./alpha);
end
if ~isempty(Fdlp)
  % double-layer potential due to all other vesicles
  valPos = valPos - Fdlp*diag(1./alpha);
end

if o.confined
  if ~isempty(Fwall2Ves)    
    % velocity due to solid walls evaluated on vesicles 
    valPos = valPos - o.dt*Fwall2Ves*diag(1./alpha);
  end
  if ~isempty(LetsVes)
    % velocity on vesicles due to the rotlets and stokeslets
    valPos = valPos - o.dt*LetsVes*diag(1./alpha);
  end
end
% END OF EVALUATING VELOCITY ON VESICLES

% START OF EVALUATING VELOCITY ON WALLS
% evaluate velocity on solid walls due to the density function.
% self solid wall interaction
if o.confined
  if o.matFreeWalls
    potWall = o.opWall;
    valWalls = valWalls - 1/2*etaM + ...
      potWall.exactStokesDLdiag(walls,o.wallDLP,etaM);
    valWalls(:,1) = valWalls(:,1) + ...
      potWall.exactStokesN0diag(walls,o.wallN0,etaM(:,1));
  end
end

if o.confined
  if ~isempty(FSLPwall)  
    % velocity on walls due to the vesicle traction jump
    valWalls = valWalls + FSLPwall;
  end
  if ~isempty(FDLPwall)
    % velocity on walls due to the vesicle viscosity jump
    valWalls = valWalls + FDLPwall/o.dt;
  end
  if o.matFreeWalls
    if ~isempty(FDLPwall2wall)  
      % velocity on walls due to all other walls
      valWalls = valWalls + FDLPwall2wall;    
    end
    if ~isempty(LetsWalls)
      % velocity on walls due to the rotlets and stokeslets
      valWalls = valWalls + LetsWalls;  
    end
  else
    if ~isempty(FDLPwall2wall) 
      valWalls = valWalls(:) + FDLPwall2wall;  
    end
  end
end % if o.confined
% END OF EVALUATING VELOCITY ON WALLS

% START OF EVALUATING INEXTENSIBILITY CONDITION
% Two possible discretizations of the inextensibility condition

% compute surface divergence of the current GMRES iterate
% method1 sets this equal to the surface divergence of
% the previous time step
valTen = vesicle.surfaceDiv(Xm);

% END OF EVALUATING INEXTENSIBILITY CONDITION

% beta times solution coming from time derivative
valPos = valPos + Xm;

% Initialize output from vesicle and inextensibility equations to zero
val = zeros(3*N*nv,1);

% Stack val as [x-coordinate;ycoordinate;tension] repeated
% nv times for each vesicle
for k=1:nv
  val((k-1)*3*N+1:3*k*N) = [valPos(:,k);valTen(:,k)];
end
if (any(vesicle.viscCont ~= 1) && o.confined)
  % This combination of options causes problems with
  % the scaling of the preconditioner.  Need to
  % get rid of the potentially small value o.dt
  valWalls = valWalls * o.dt;
end

% Stack velocity along the solid walls in same manner as above
% Stack the stokeslets and rotlet componenets at the end
if o.confined
  val = [val;valWalls(:);valLets];
end
  
end % TimeMatVec


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function z = letsIntegrals(o,otlets,etaM)
% z = letsIntegrals(o,otlets,etaM,etaMint,walls,wallsInt) integrates 
% the density function to enforce constraints on stokeslets and rotlets
% if there are walls with different discretizations (i.e. one large outer
% wall and multiple small inner walls), then takes only the inner walls)
walls = o.walls;
Nbd = walls.N;
nvbd = walls.nv;

z = zeros(3*(nvbd-1),1);

for k = 2:nvbd
  % two stokeslet terms per inner boundary  
  stokeslet = otlets(3*(k-2)+1:3*(k-2)+2);
  % one rotlet term per inner boundary
  rotlet = otlets(3*(k-1));
  % integral of density function dotted with [1;0]
  % is one stokeslet
  ind = 3*(k-2)+1;
  z(ind) = -2*pi*stokeslet(1) + ...
    sum(etaM(1:Nbd,k).*walls.sa(:,k))*2*pi/Nbd;
  % integral of density fuction dotted with [0;1]
  % is the other stokeslet
  z(ind+1) = -2*pi*stokeslet(2) + ...
    sum(etaM(Nbd+1:2*Nbd,k).*walls.sa(:,k))*2*pi/Nbd;
  % integral of density function dotted with (-y,x)
  % is the rotlet
  z(ind+2) = -2*pi*rotlet + sum(...
    ((walls.X(Nbd+1:2*Nbd,k)).*etaM(1:Nbd,k) - ...
    (walls.X(1:Nbd,k)).*etaM(Nbd+1:2*Nbd,k)).*...
    walls.sa(:,k))*2*pi/Nbd;
end % k


end % letsIntegrals

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function vel = RSlets(o,X,center,stokeslet,rotlet)
% vel = RSlets(o,X,center,stokeslet,rotlet) evaluates the velocity due
% to the stokeslet and rotlet terms.  Center of the rotlet and
% stokeslet is contained in center

oc = curve;
% set of points where we are evaluating the velocity
[x,y] = oc.getXY(X);
% the center of the rotlet/stokeslet terms
[cx,cy] = oc.getXY(center);

% distance squared
rho2 = (x-cx).^2 + (y-cy).^2;

% x component of velocity due to the stokeslet and rotlet
LogTerm = -0.5*log(rho2)*stokeslet(1);
rorTerm = 1./rho2.*((x-cx).*(x-cx)*stokeslet(1) + ...
    (x-cx).*(y-cy)*stokeslet(2));
RotTerm = (y-cy)./rho2*rotlet;
velx = 1/4/pi*(LogTerm + rorTerm) + RotTerm;

% y component of velocity due to the stokeslet and rotlet
LogTerm = -0.5*log(rho2)*stokeslet(2);
rorTerm = 1./rho2.*((y-cy).*(x-cx)*stokeslet(1) + ...
    (y-cy).*(y-cy)*stokeslet(2));
RotTerm = -(x-cx)./rho2*rotlet;
vely = 1/4/pi*(LogTerm + rorTerm) + RotTerm;

% velocity
vel = [velx;vely];


end % RSlets

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% START OF DIFFERENT PRECONDITIONERS INCLUDING BLOCK-DIAGONAL, ONE FOR
% THE SYSTEM THAT SOLVES FOR THE TENSION AND DENSITY GIVEN A POSITION,
% MULTIGRID IDEAS, SCHUR COMPLEMENTS, AND ANALYTIC (BASED ON A
% CIRCLE).  THE BLOCK-DIAGONAL PRECONDITIONER IS THE MOST ROBUST
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function val = preconditionerBD(o,z)
% val = preconditionBD(z) applies the block diagonal preconditioner
% required by preconditioned-GMRES to the vector z


nv = size(o.bdiagVes.L,3); % number of vesicles
N = size(o.bdiagVes.L,1)/3; % number of points


% extract the position and tension part.  Solid walls is
% handled in the next section of this routine
zves = z(1:3*N*nv);

valVes = zeros(3*N*nv,1);
% precondition with the block diagonal preconditioner for the
  % vesicle position and tension

for k=1:nv
  valVes((k-1)*3*N+1:3*k*N) = o.bdiagVes.U(:,:,k)\...
    (o.bdiagVes.L(:,:,k)\zves((k-1)*3*N+1:3*k*N));
end % k
  

% part of z from solid walls
zwalls = z(3*N*nv+1:end);


% if not out-of-core, then the inverse is on the memory      
valWalls = o.bdiagWall * zwalls;


% stack the two componenets of the preconditioner
val = [valVes;valWalls];


end % preconditionerBD

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Mat = wallsPrecond(o)
% wallsPrecond computes the matrix which is the 
% exact inverse of
% the double-layer potential for stokes flow in a bounded domain.  Used
% in the preconditioner for vesicle simulations and capsules.m/computeEta
% which computes eta and RS when there is no vesicle.

walls = o.walls;
Nbd = walls.N;
nvbd = walls.nv;
oc = curve;
[x,y] = oc.getXY(walls.X);
[nory,norx] = oc.getXY(walls.xt);
nory = -nory;
sa = walls.sa;
[cx,cy] = oc.getXY(walls.center);

% Allocate space for blocks of matrix that carries the double- layer
% potential, rotlets, and stokeslets to the velocity and the conditions
% in (A4) and (A5) in Rahimian et al.
M11 = zeros(2*Nbd*nvbd,2*Nbd*nvbd);
M12 = zeros(2*Nbd*nvbd,3*(nvbd-1));
M21 = zeros(3*(nvbd-1),2*Nbd*nvbd);


% Self interaction terms with the jump coming from the double layer
% potential
M11(1:2*Nbd,1:2*Nbd) = M11(1:2*Nbd,1:2*Nbd) + o.wallN0(:,:,1);
jump = - 1/2; 
for k = 1:nvbd
istart = (k-1)*2*Nbd+1;
iend = 2*k*Nbd;
M11(istart:iend,istart:iend) = M11(istart:iend,istart:iend) + ...
    jump*eye(2*Nbd) + o.wallDLP(:,:,k);
end


for ktar = 1:nvbd % loop over targets
itar = 2*(ktar-1)*Nbd + 1;
jtar = 2*ktar*Nbd;
K = [(1:ktar-1) (ktar+1:nvbd)];

D = zeros(2*Nbd,2*Nbd);
for ksou = K % loop over all other walls
  isou = 2*(ksou-1)*Nbd + 1;
  jsou = 2*ksou*Nbd;

  xtar = x(:,ktar); ytar = y(:,ktar);
  xtar = xtar(:,ones(Nbd,1)); 
  ytar = ytar(:,ones(Nbd,1));

  xsou = x(:,ksou); ysou = y(:,ksou);
  xsou = xsou(:,ones(Nbd,1))';
  ysou = ysou(:,ones(Nbd,1))';

  norxtmp = norx(:,ksou); norytmp = nory(:,ksou);
  norxtmp = norxtmp(:,ones(Nbd,1))';
  norytmp = norytmp(:,ones(Nbd,1))';

  satmp = sa(:,ksou);
  satmp = satmp(:,ones(Nbd,1))';

  rho2 = (xtar-xsou).^2 + (ytar-ysou).^2;

  coeff = 1/pi*((xtar-xsou).*norxtmp + ...
      (ytar-ysou).*norytmp).*satmp./rho2.^2;

  D(1:Nbd,:) = 2*pi/Nbd*[coeff.*(xtar-xsou).^2 ...
      coeff.*(xtar-xsou).*(ytar-ysou)];
  D(Nbd+1:end,:) = 2*pi/Nbd*[coeff.*(ytar-ysou).*(xtar-xsou) ...
      coeff.*(ytar-ysou).^2];

  M11(itar:jtar,isou:jsou) = D;
end %end ktar
end %end ksou

% These compute the integral of the density function around each of the
% inner componenents of the geometry
for k = 1:nvbd-1
icol = 3*(k-1)+1;
istart = 2*k*Nbd+1;
iend = istart + Nbd - 1;
M21(icol,istart:iend) = 2*pi/Nbd*sa(:,k+1)';
M21(icol+2,istart:iend) = 2*pi/Nbd*sa(:,k+1)'.*y(:,k+1)';
istart = istart + Nbd;
iend = iend + Nbd;
M21(icol+1,istart:iend) = 2*pi/Nbd*sa(:,k+1)';
M21(icol+2,istart:iend) = -2*pi/Nbd*sa(:,k+1)'.*x(:,k+1)';
end % k


% This is the evaluation of the velocity field due to the stokeslet
% and rotlet terms
for k = 1:nvbd - 1
for ktar = 1:nvbd
  rho2 = (x(:,ktar) - cx(k+1)).^2 + (y(:,ktar) - cy(k+1)).^2;
  istart = (ktar-1)*2*Nbd + 1;
  iend = istart + Nbd - 1;

  icol = 3*(k-1)+1;
  M12(istart:iend,icol) = ...
    M12(istart:iend,icol) + ...
    1/4/pi*(-0.5*log(rho2) + (x(:,ktar)-cx(k+1))./rho2.*...
        (x(:,ktar)-cx(k+1)));
  M12(istart + Nbd:iend + Nbd,icol) = ...
    M12(istart + Nbd:iend + Nbd,icol) + ...
    1/4/pi*((x(:,ktar)-cx(k+1))./rho2.*(y(:,ktar)-cy(k+1)));

  icol = 3*(k-1)+2;
  M12(istart:iend,icol) = ...
    M12(istart:iend,icol) + ...
    1/4/pi*((y(:,ktar)-cy(k+1))./rho2.*(x(:,ktar)-cx(k+1)));
  M12(istart + Nbd:iend + Nbd,icol) = ...
    M12(istart + Nbd:iend + Nbd,icol) + ...
    1/4/pi*(-0.5*log(rho2) + (y(:,ktar)-cy(k+1))./rho2.*...
        (y(:,ktar)-cy(k+1)));

  icol = 3*(k-1)+3;
  M12(istart:iend,icol) = ...
    M12(istart:iend,icol) + ...
    (y(:,ktar)-cy(k+1))./rho2;
  M12(istart + Nbd:iend + Nbd,icol) = ...
    M12(istart + Nbd:iend + Nbd,icol) - ...
    (x(:,ktar)-cx(k+1))./rho2;
end
end

% different combinations of the density functions have to multiply to
% 2*pi multiplied by rotlet or stokeslet terms
M22 = -2*pi*eye(3*(nvbd-1));

% Save the wall2wall interaction matrices if not matrix free
if ~o.matFreeWalls
o.wallDLPandRSmat = [M11 M12; M21 M22];
end

% invert the matrix
Mat = ([M11 M12; M21 M22])\eye(2*nvbd*Nbd + 3*(nvbd-1));

    
end % wallsPrecond

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function vInf = bgFlow(o,X,Xwalls,varargin)
% [vInf = bgFlow(o,X,Xwalls,varargin) computes the velocity field at the 
% points X. vInf is either background or wall velocity. Flows are given by:
%     relaxation:     (0,0)
%     extensional:    (-x,y)
%     parabolic:      (k(1-y/r)^2,0)
%     taylorGreen:    (sin(x)cos(y),-cos(x)sin(y))
%     shear:          (ky,0)
%     choke:          poeusille-like flow at intake and outtake
%     doublechoke:    same as choke
%     couette:        rotating boundary
%     doubleCouette   two rotating boundaries
%     tube            poiseuille flow in a tube 



% the farfield velocity is vInf -- it is either free-space and defined on
% the vesicles or confined flow and defined on Xwalls


N = size(X,1)/2; % number of points per vesicle
nv = size(X,2); % number of vesicles

oc = curve;

% Separate out x and y coordinates of vesicles
[x,y] = oc.getXY(X);

% speed of the background velocity
speed = varargin{find(strcmp(varargin,'Speed'))+1};    


if any(strcmp(varargin,'relaxation'))
  vInf = zeros(2*N,nv); 

elseif any(strcmp(varargin,'extensional'))
  vInf = [-x;y];

elseif any(strcmp(varargin,'parabolic'))
  chanWidth = varargin{find(strcmp(varargin,'chanWidth'))+1};    
  vInf = [(1-(y/chanWidth).^2);zeros(N,nv)];

elseif any(strcmp(varargin,'taylorGreen'))
  vortexSize = varargin{find(strcmp(varargin,'vortexSize'))+1};      
  vInf = vortexSize*[sin(x/vortexSize * pi).*cos(y/vortexSize * pi);-cos(x/vortexSize * pi).*sin(y/vortexSize * pi)];

elseif any(strcmp(varargin,'shear'))
  vInf = [y;zeros(N,nv)];
  
elseif (any(strcmp(varargin,'choke')) || ...
      any(strcmp(varargin,'doublechoke')) || ...
      any(strcmp(varargin,'choke2')))
  % this one assumes one external wall
  xwalls = Xwalls(1:end/2,1); ywalls = Xwalls(end/2+1:end,1);
  Nbd = numel(xwalls);
  vInf = zeros(2*Nbd,1);
  ind = abs(xwalls)>0.8*max(xwalls);
  vx = exp(1./((ywalls(ind)/max(ywalls)).^2-1))/exp(-1);
  % typical mollifer so that velocity decays smoothly to 0
  vx(vx==Inf) = 0;
  vInf(ind,:) = vx;
elseif (any(strcmp(varargin,'tube')))
  xwalls = Xwalls(1:end/2,1); ywalls = Xwalls(end/2+1:end,1);
  Nbd = numel(xwalls);
  vInf = zeros(2*Nbd,1);
  ind = abs(xwalls)>0.8*max(xwalls);
  vx = exp(1./((ywalls(ind)/max(ywalls)).^2-1))/exp(-1);
  % typical mollifer so that velocity decays smoothly to 0
  vx(vx==Inf) = 0;
  vInf(ind,:) = vx;

elseif any(strcmp(varargin,'couette'));
  % there are several walls in this one
  xwalls = Xwalls(1:end/2,1); ywalls = Xwalls(end/2+1:end,1);
  Nbd = numel(xwalls);  
  vInf = [zeros(2*Nbd,1) 1*[-ywalls(:,2)+mean(ywalls(:,2));xwalls(:,2)-mean(xwalls(:,2))]];
  
elseif any(strcmp(varargin,'doubleCouette'))
  xwalls = Xwalls(1:end/2,1); ywalls = Xwalls(end/2+1:end,1);
  Nbd = numel(xwalls);
  vInf = [zeros(2*Nbd,1) 1*[-ywalls(:,2)+mean(ywalls(:,2));xwalls(:,2)-mean(xwalls(:,2))] ...
      -[ywalls(:,3)-mean(ywalls(:,3));-xwalls(:,3)+mean(xwalls(:,3))]];

  
end

% Scale the velocity
vInf = vInf * speed;

end % bgFlow

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end % methods

end % classdef
