classdef poten_py

properties
  N; % points per curve
  Nup; 
  Nquad;
  qw; % quadrature weights for logarithmic singularity
  qp; % quadrature points for logarithmic singularity (Alpert's rule)
  
  % upsampled quadrature rules for Alpert's quadrature rule.
  Rfor;
  Rbac;
  
  Prolong_LP;
  Restrict_LP;
  % prolongation and restriction matrices for layer potentitals

  interpMat; % interpolation matrix used for near-singular integration

end % properties

methods

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function o = poten_py(N)
% o = poten(N,Nup): constructor; N is the number of points per
% curve; Nup is the number of points on an upsampled grid that is used to remove antialiasing.  initialize class.


o.N = N;
o.Nup = N*ceil(sqrt(o.N));

[o.qw, o.qp, o.Rbac, o.Rfor] = o.singQuadStokesSLmatrix(o.Nup);
o.Nquad = numel(o.qw);
o.qw = o.qw(:,ones(o.Nup,1));

o.interpMat = o.lagrangeInterp;

[o.Restrict_LP, o.Prolong_LP] = fft1_py.fourierRandP(o.N,o.Nup); % you already have this one
end % poten: constructor
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function G = stokesSLmatrix(o,vesicle)


x = interpft(vesicle.X(1:end/2,:),o.Nup);
y = interpft(vesicle.X(end/2+1:end,:),o.Nup);
Xup = [x; y];
vesicleUp = capsules_py(Xup,[],[],1,ones(vesicle.nv,1));


G = zeros(2*vesicle.N,2*vesicle.N,vesicle.nv);
for k=1:vesicle.nv  % Loop over curves
  xx = x(:,k);
  yy = y(:,k);
  % locations
  sa = vesicleUp.sa(:,k)';
  sa = sa(ones(vesicleUp.N,1),:);
  % Jacobian

  xtar = xx(:,ones(o.Nquad,1))'; 
  ytar = yy(:,ones(o.Nquad,1))'; 
  % target points

  xsou = xx(:,ones(vesicleUp.N,1)); 
  ysou = yy(:,ones(vesicleUp.N,1));
  % source points
  
  xsou = xsou(o.Rfor);
  ysou = ysou(o.Rfor);
  % have to rotate each column so that it is compatiable with o.qp
  % which is the matrix that takes function values and maps them to the
  % intermediate values required for Alpert quadrature
  
  diffx = xtar - o.qp*xsou;
  diffy = ytar - o.qp*ysou;
  rho2 = (diffx.^2 + diffy.^2).^(-1);
  % one over distance squared

  logpart = 0.5*o.qp'*(o.qw .* log(rho2));
  % sign changed to positive because rho2 is one over distance squared

  Gves = logpart + o.qp'*(o.qw.*diffx.^2.*rho2);
  Gves = Gves(o.Rbac);
  G11 = Gves'.*sa;
  % (1,1)-term
 
  Gves = logpart + o.qp'*(o.qw.*diffy.^2.*rho2);
  Gves = Gves(o.Rbac);
  G22 = Gves'.*sa;
  % (2,2)-term

  Gves = o.qp'*(o.qw.*diffx.*diffy.*rho2);
  Gves = Gves(o.Rbac);
  G12 = Gves'.*sa;
  % (1,2)-term
  
  
  G(1:o.N,1:o.N,k) = o.Restrict_LP * G11 * o.Prolong_LP;
  G(1:o.N,o.N+1:end,k) = o.Restrict_LP * G12 * o.Prolong_LP;
  G(o.N+1:end,1:o.N,k) = G(1:o.N,o.N+1:end,k);
  G(o.N+1:end,o.N+1:end,k) = o.Restrict_LP * G22 * o.Prolong_LP;

end

end % stokesSLmatrix

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function D = stokesDLmatrix(o,vesicle)
% D = stokesDLmatrix(vesicle), generate double-layer potential for 
% Stokes vesicle is a data structure defined as in the capsules class
% D is (2N,2N,nv) array where N is the number of points per curve and 
% nv is the number of curves in X 

x = interpft(vesicle.X(1:end/2,:),o.Nup);
y = interpft(vesicle.X(end/2+1:end,:),o.Nup);
Xup = [x; y];
vesicleUp = capsules_py(Xup,[],[],1,ones(vesicle.nv,1));

% initialize space for double-layer potential matrix
D = zeros(2*vesicle.N,2*vesicle.N,vesicle.nv);
for k=1:vesicle.nv  % Loop over curves
  if (vesicle.viscCont(k) ~= 1)
    const_coeff = -(1-vesicle.viscCont(k));
    % constant that has the d\theta and scaling with the viscosity
    % contrast
    xx = x(:,k);
    yy = y(:,k);
    % locations

    
    % upsampled single versicle
    tx = vesicleUp.xt(1:end/2,k);
    ty = vesicleUp.xt(end/2+1:end,k);
    
    % Vesicle tangent
    sa = vesicleUp.sa(:,k)';
    % Jacobian
    cur = vesicleUp.cur(:,k)';
    

    xtar = xx(:,ones(vesicleUp.N,1))';
    ytar = yy(:,ones(vesicleUp.N,1))';
    % target points

    xsou = xx(:,ones(vesicleUp.N,1));
    ysou = yy(:,ones(vesicleUp.N,1));
    % source points

    txsou = tx';
    tysou = ty';
    % tangent at srouces
    sa = sa(ones(vesicleUp.N,1),:);
    % Jacobian

    diffx = xtar - xsou;
    diffy = ytar - ysou;
    rho4 = (diffx.^2 + diffy.^2).^(-2);
    rho4(1:vesicleUp.N+1:vesicleUp.N^2) = 0;
    % set diagonal terms to 0

    kernel = diffx.*(tysou(ones(vesicleUp.N,1),:)) - ...
            diffy.*(txsou(ones(vesicleUp.N,1),:));
    kernel = kernel.*rho4.*sa;
    kernel = const_coeff*kernel;

    D11 = kernel.*diffx.^2;
    % (1,1) component
    D11(1:vesicleUp.N+1:vesicleUp.N^2) = 0.5*const_coeff*cur.*sa(1,:).*txsou.^2;
    % diagonal limiting term

    D12 = kernel.*diffx.*diffy;
    % (1,2) component
    D12(1:vesicleUp.N+1:vesicleUp.N^2) = 0.5*const_coeff*cur.*sa(1,:).*txsou.*tysou;
    % diagonal limiting term

    D22 = kernel.*diffy.^2;
    % (2,2) component
    D22(1:vesicleUp.N+1:vesicleUp.N^2) = 0.5*const_coeff*cur.*sa(1,:).*tysou.^2;
    % diagonal limiting term

    
    D11 = o.Restrict_LP * D11 * o.Prolong_LP; 
    D12 = o.Restrict_LP * D12 * o.Prolong_LP; 
    D22 = o.Restrict_LP * D22 * o.Prolong_LP; 
    
    % move to grid with vesicle.N points by applying prolongation and
    % restriction operators

    D(:,:,k) = [D11 D12; D12 D22];
    % build matrix with four blocks
    D(:,:,k) = 1/pi*D(:,:,k)*2*pi/vesicleUp.N;
    % scale with the arclength spacing and divide by pi

  end
end % k

end % stokesDLmatrix

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function N0 = stokesN0matrix(o,vesicle)
% N0 = stokesN0matrix(vesicle) generates the the integral operator with kernel
% normal(x) \otimes normal(y) which removes the rank one defficiency of the
% double-layer potential.  Need this operator for solid walls

oc = curve_py;
[x,y] = oc.getXY(vesicle.X); % Vesicle positions

normal = [vesicle.xt(vesicle.N+1:2*vesicle.N,:);...
         -vesicle.xt(1:vesicle.N,:)]; % Normal vector
normal = normal(:,ones(2*vesicle.N,1));

sa = [vesicle.sa(:,1);vesicle.sa(:,1)];
sa = sa(:,ones(2*vesicle.N,1));
N0 = zeros(2*vesicle.N,2*vesicle.N,vesicle.nv);
N0(:,:,1) = normal.*normal'.*sa'*2*pi/vesicle.N;
% Use N0 if solving (-1/2 + DLP)\eta = f where f has no flux through
% the boundary.  By solving (-1/2 + DLP + N0)\eta = f, we guarantee
% that \eta also has no flux through the boundary.  This is not
% required, but it means we're enforcing one addition condition on eta
% which removes the rank one kernel.  DLP is the double-layer potential
% for stokes equation

end % stokesN0matrix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% START OF ROUTINES THAT EVALUATE LAYER-POTENTIALS
% WHEN SOURCES == TARGETS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function SLP = exactStokesSLdiag(o,vesicle,G,f)
% SLP = exactStokesSLdiag(vesicle,G,f) computes the diagonal term of
% the single-layer potential due to f around vesicle.  Source and
% target points are the same.  This uses Alpert's quadrature formula.

SLP = zeros(2*vesicle.N,vesicle.nv);
for k = 1:vesicle.nv
  SLP(:,k) = G(:,:,k) * f(:,k);
end


end % exactStokesSLdiag

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function DLP = exactStokesDLdiag(o,vesicle,D,f)
% DLP = exactStokesDLdiag(vesicle,f,K) computes the diagonal term of
% the double-layer potential due to f around all vesicles.  Source and
% target points are the same.  This uses trapezoid rule with the
% curvature at the diagonal in order to guarantee spectral accuracy.


DLP = zeros(2*vesicle.N,vesicle.nv);
for k = 1:vesicle.nv
  DLP(:,k) = D(:,:,k) * f(:,k);
end


end % exactStokesDLdiag

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function N0 = exactStokesN0diag(o,vesicle,N0,f)
% DLP = exactStokesN0diag(vesicle,f) computes the diagonal term of the
% modification of the double-layer potential due to f around outermost
% vesicle.  Source and target points are the same.  This uses trapezoid
% rule
if isempty(N0)
  N = size(f,1)/2;
  oc = curve_py;
  [fx,fy] = oc.getXY(f(:,1));
  fx = fx.*vesicle.sa(:,1);
  fy = fy.*vesicle.sa(:,1);
  [tx,ty] = oc.getXY(vesicle.xt(:,1));
  % tangent vector
  const = sum(ty.*fx - tx.*fy)*2*pi/N;
  % function to be integrated is dot product of normal with density
  % function
  N0 = zeros(2*N,1);
  N0 = const*[ty;-tx];
else
  N0 = N0(:,:,1)*f(:,1);
end

end % exactStokesN0diag

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% END OF ROUTINES THAT EVALUATE LAYER-POTENTIALS
% WHEN SOURCES == TARGETS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% START OF ROUTINES THAT EVALUATE LAYER-POTENTIALS
% WHEN SOURCES ~= TARGETS.  CAN COMPUTE LAYER POTENTIAL ON EACH
% VESICLE DUE TO ALL OTHER VESICLES (ex. stokesSLP) AND CAN
% COMPUTE LAYER POTENTIAL DUE TO VESICLES INDEXED IN K1 AT 
% TARGET POINTS Xtar
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function stokesSLPtar = exactStokesSL(o,vesicle,f,Xtar,K1)
% [stokesSLP,stokesSLPtar] = exactStokesSL(vesicle,f,Xtar,K1) computes
% the single-layer potential due to f around all vesicles except
% itself.  Also can pass a set of target points Xtar and a collection
% of vesicles K1 and the single-layer potential due to vesicles in K1
% will be evaluated at Xtar.  Everything but Xtar is in the 2*N x nv
% format Xtar is in the 2*Ntar x ncol format

if nargin == 5
  Ntar = size(Xtar,1)/2;
  ncol = size(Xtar,2);
  stokesSLPtar = zeros(2*Ntar,ncol);
else
  K1 = [];
  Ntar = 0;
  stokesSLPtar = [];
  ncol = 0;
  % if nargin ~= 5, the user does not need the velocity at arbitrary
  % points
end

den = f.*[vesicle.sa;vesicle.sa]*2*pi/vesicle.N;
% multiply by arclength term

xsou = vesicle.X(1:end/2,K1);
ysou = vesicle.X(end/2+1:end,K1);
xsou = xsou(:); ysou = ysou(:);
xsou = xsou(:,ones(Ntar,1))';
ysou = ysou(:,ones(Ntar,1))';
% This is faster than repmat

denx = den(1:end/2,K1);
deny = den(end/2+1:end,K1);
denx = denx(:); deny = deny(:);
denx = denx(:,ones(Ntar,1))';
deny = deny(:,ones(Ntar,1))';
% This is faster than repmat

for k = 1:ncol % loop over columns of target points 
  xtar = Xtar(1:end/2,k); ytar = Xtar(end/2+1:end,k);
  xtar = xtar(:,ones(vesicle.N*numel(K1),1));
  ytar = ytar(:,ones(vesicle.N*numel(K1),1));
  
  diffx = xtar-xsou; diffy = ytar-ysou;
  
  dis2 = diffx.^2 + diffy.^2;
  % distance squared of source and target location
  
  coeff = 0.5*log(dis2);
  % first part of single-layer potential for Stokes
  stokesSLPtar(1:Ntar,k) = -sum(coeff.*denx,2);
  stokesSLPtar(Ntar+1:2*Ntar,k) = -sum(coeff.*deny,2);
  % log part of stokes single-layer potential

  coeff = (diffx.*denx + diffy.*deny)./dis2;
  % second part of single-layer potential for Stokes
  stokesSLPtar(1:Ntar,k) = stokesSLPtar(1:Ntar,k) + ...
      sum(coeff.*diffx,2);
  stokesSLPtar(Ntar+1:2*Ntar,k) = stokesSLPtar(Ntar+1:2*Ntar,k) + ...
      sum(coeff.*diffy,2);
end
stokesSLPtar = 1/(4*pi)*stokesSLPtar;
% Avoid loop over the target points.  Only loop over its columns
% 1/4/pi is the coefficient in front of the single-layer potential

end % exactStokesSL
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [laplaceDLPtar] = exactLaplaceDL(o,vesicle,f,Xtar,K1)
% pot = exactLaplaceDL(vesicle,f,Xtar,K1) computes the double-layer
% laplace potential due to f around all vesicles except itself.  Also
% can pass a set of target points Xtar and a collection of vesicles K1
% and the double-layer potential due to vesicles in K1 will be
% evaluated at Xtar.  Everything but Xtar is in the 2*N x nv format
% Xtar is in the 2*Ntar x ncol format

oc = curve_py;

nx = vesicle.xt(vesicle.N+1:2*vesicle.N,:);
ny = -vesicle.xt(1:vesicle.N,:);

Ntar = size(Xtar,1)/2;
ncol = size(Xtar,2);
laplaceDLPtar = zeros(2*Ntar,ncol);

den = f.*[vesicle.sa;vesicle.sa]*2*pi/vesicle.N;
% multiply by arclength term

[xsou,ysou] = oc.getXY(vesicle.X(:,K1));
xsou = xsou(:); ysou = ysou(:);
xsou = xsou(:,ones(Ntar,1))';
ysou = ysou(:,ones(Ntar,1))';

[denx,deny] = oc.getXY(den(:,K1));
denx = denx(:); deny = deny(:);
denx = denx(:,ones(Ntar,1))';
deny = deny(:,ones(Ntar,1))';

nxK1 = nx(:,K1); nyK1 = ny(:,K1);
nxK1 = nxK1(:); nyK1 = nyK1(:);
nxK1 = nxK1(:,ones(Ntar,1))';
nyK1 = nyK1(:,ones(Ntar,1))';

for k2 = 1:ncol % loop over columns of target points
  [xtar,ytar] = oc.getXY(Xtar(:,k2));
  xtar = xtar(:,ones(vesicle.N*numel(K1),1));
  ytar = ytar(:,ones(vesicle.N*numel(K1),1));
  
  diffx = xsou-xtar; diffy = ysou-ytar;
  dis2 = diffx.^2 + diffy.^2;
  
  coeff = (diffx.*nxK1 + diffy.*nyK1)./dis2;
  
  val = coeff.*denx;
  laplaceDLPtar(1:Ntar,k2) = sum(val,2);
  
  val = coeff.*deny;
  laplaceDLPtar(Ntar+1:2*Ntar,k2) = sum(val,2);
end % end k2
% Evaluate double-layer potential at arbitrary target points
laplaceDLPtar = 1/(2*pi)*laplaceDLPtar;
% 1/2/pi is the coefficient in front of the double-layer potential

end % exactLaplaceDL

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function stokesDLPtar = exactStokesDL(o,vesicle,f,Xtar,K1)
% [stokesDLP,stokesDLPtar] = exactStokesDL(vesicle,f,Xtar,K1) computes
% the double-layer potential due to f around all vesicles except
% itself.  Also can pass a set of target points Xtar and a collection
% of vesicles K1 and the double-layer potential due to vesicles in K1
% will be evaluated at Xtar.  Everything but Xtar is in the 2*N x nv
% format Xtar is in the 2*Ntar x ncol format

normal = [vesicle.xt(vesicle.N+1:2*vesicle.N,:);...
         -vesicle.xt(1:vesicle.N,:)]; 
% Normal vector

if nargin == 5
  Ntar = size(Xtar,1)/2;
  ncol = size(Xtar,2);
  stokesDLPtar = zeros(2*Ntar,ncol);
else
  K1 = [];
  stokesDLPtar = [];
  ncol = 0;
  Ntar = 0;
  % if nargin ~= 5, the user does not need the velocity at arbitrary
  % points
end

den = (f.*[vesicle.sa;vesicle.sa]*2*pi/vesicle.N)* diag(1-vesicle.viscCont);
% jacobian term and 2*pi/N accounted for here
% have accounted for the scaling with (1-\nu) here

oc = curve_py;
[xsou,ysou] = oc.getXY(vesicle.X(:,K1));
xsou = xsou(:); ysou = ysou(:);
xsou = xsou(:,ones(Ntar,1))';
ysou = ysou(:,ones(Ntar,1))';

[denx,deny] = oc.getXY(den(:,K1));
denx = denx(:); deny = deny(:);
denx = denx(:,ones(Ntar,1))';
deny = deny(:,ones(Ntar,1))';

[normalx,normaly] = oc.getXY(normal(:,K1));
normalx = normalx(:); normaly = normaly(:);
normalx = normalx(:,ones(Ntar,1))';
normaly = normaly(:,ones(Ntar,1))';

for k = 1:ncol % loop over columns of target points
  [xtar,ytar] = oc.getXY(Xtar(:,k));
  xtar = xtar(:,ones(vesicle.N*numel(K1),1));
  ytar = ytar(:,ones(vesicle.N*numel(K1),1));
  
  diffx = xtar-xsou; diffy = ytar-ysou;
  dis2 = (diffx).^2 + (diffy).^2;
  % difference of source and target location and distance squared
  
  
  rdotnTIMESrdotf = (diffx.*normalx + diffy.*normaly)./dis2.^2 .* ...
      (diffx.*denx + diffy.*deny);
  % \frac{(r \dot n)(r \dot density)}{\rho^{4}} term
  
  stokesDLPtar(1:Ntar,k) = stokesDLPtar(1:Ntar,k) + ...
      sum(rdotnTIMESrdotf.*diffx,2);
  stokesDLPtar(Ntar+1:end,k) = stokesDLPtar(Ntar+1:end,k) + ...
      sum(rdotnTIMESrdotf.*diffy,2);
  % r \otimes r term of the double-layer potential
end
stokesDLPtar = stokesDLPtar/pi;
% double-layer potential due to vesicles indexed over K1 evaluated at
% arbitrary points


end % exactStokesDL

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% END OF ROUTINES THAT EVALUATE LAYER-POTENTIALS
% WHEN SOURCES ~= TARGETS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function LP = nearSingInt(o,vesicleSou,f,selfMat,...
    NearStruct,kernelDirect,vesicleTar,tEqualS)
% 
% vesicleSou : Source points
% f : density
% selfMat: function selfMat(f) : self-interactions
% NearStruct : output of getZone()
% kernelDirect: direct kernel evaluations between source and target points
% tEqualsS: source and target points coincide
% near zone
%
% LP =
% nearSingInt(vesicle,f,selfMat,NearStruct,kernelDirect,vesicleTar,tEqualS)
% computes a layer potential due to f at all points in vesicleTar.X.  If
% tEqualS==true, then the vesicleTar == vesicleSou and the self-vesicle
% interaction is skipped.  selfMat is the diagonal of the potential needed to
% compute the layer potential of each vesicle indepenedent of all others.
% kernel and kernelDirect are two (possibly the same) routines that compute
% the layer potential.  kernelDirect always uses the direct method whereas
% kernel may use an FMM-accelerated method.  NearStruct is a structure
% containing the variables zone,dist,nearest,icp,argnear which are required by
% near-singular integration (they keep everything sorted and precomputed)
% Everything is in the 2*N x nv format Can pass a final argument if desired so
% that plots of the near-singular integration algorithm are displayed

if (tEqualS && size(vesicleSou.X,2) == 1)
  LP = zeros(size(vesicleSou.X));
  return
end
% only a single vesicle, so velocity on all other vesicles will always
% be zero

% GOKBERK: FOR THIS ONE YOU SHOULD USE YOUR NEAR-ZONE STRUCTURE CODE
dist = NearStruct.dist;
zone = NearStruct.zone;
nearest = NearStruct.nearest;
icp = NearStruct.icp;
argnear = NearStruct.argnear;

Xsou = vesicleSou.X; % source positions
Nsou = size(Xsou,1)/2; % number of source points
nvSou = size(Xsou,2); % number of source 'vesicles'
Xtar = vesicleTar.X; % target positions
Ntar = size(Xtar,1)/2; % number of target points
nvTar = size(Xtar,2); % number of target 'vesicles'

h = vesicleSou.length/Nsou; % arclength term


Nup = o.Nup;

% Integral on itself
vself = selfMat(f);

% upsample to Nup  
Xup = [interpft(Xsou(1:Nsou,:),Nup);...
       interpft(Xsou(Nsou+1:2*Nsou,:),Nup)];
fup = [interpft(f(1:Nsou,:),Nup);...
       interpft(f(Nsou+1:2*Nsou,:),Nup)];

% Compute velocity due to each vesicle independent of others.  This is
% needed when using near-singular integration since we require a value
% for the layer-potential on the vesicle of sources 

% allocate space for storing velocity at intermediate points needed
% by near-singular integration

vesicleUp = capsules_py(Xup,[],[],vesicleSou.kappa,vesicleSou.viscCont);
% Build an object with the upsampled vesicle

interpOrder = size(o.interpMat,1);
% lagrange interpolation order
p = ceil((interpOrder+1)/2);
% want half of the lagrange interpolation points to the left of the
% closest point and the other half to the right

if tEqualS % sources == targets
  if nvSou > 1
   
      
    for k = 1:nvSou
      K = [(1:k-1) (k+1:nvSou)];
      farField(:,k) = kernelDirect(vesicleUp,fup,Xtar(:,k),K);
    end
      % This is a huge savings if we are using a direct method rather
      % than the fmm to evaluate the layer potential.  The speedup is
      % more than N^{1/2}, where N is the resolution of the vesicles
      % that we are computing with
   
  else
    farField = zeros(2*Ntar,nvTar);
  end

else % sources ~= targets
  farField = kernelDirect(vesicleUp,fup,Xtar,1:nvSou);
  % evaluate layer potential due to all 'vesicles' at all points in
  % Xtar;
end
% Use upsampled trapezoid rule to compute layer potential

nearField = zeros(2*Ntar,nvTar);

beta = 1.1;
% small buffer to make sure Lagrange interpolation points are
% not in the near zone
for k1 = 1:nvSou
  if tEqualS % sources == targets
    K = [(1:k1-1) (k1+1:nvTar)];
    % skip diagonal vesicle
  else % sources ~= targets
    K = (1:nvTar);
    % consider all vesicles
  end
  
  for k2 = K
    J = find(zone{k1}(:,k2) == 1);
    % set of points on vesicle k2 close to vesicle k1
    if (numel(J) ~= 0)
      indcp = icp{k1}(J,k2);
      % closest point on vesicle k1 to each point on vesicle k2 
      % that is close to vesicle k1
      for j = 1:numel(J)
        pn = mod((indcp(j)-p+1:indcp(j)-p+interpOrder)' - 1,Nsou) + 1;
        % index of points to the left and right of the closest point
        v = filter(1,[1 -full(argnear{k1}(J(j),k2))],...
          o.interpMat*vself(pn,k1));
        vel(J(j),k2,k1) = v(end);  
        % x-component of the velocity at the closest point
        v = filter(1,[1 -full(argnear{k1}(J(j),k2))],...
          o.interpMat*vself(pn+Nsou,k1));
        vel(J(j)+Ntar,k2,k1) = v(end);
        % y-component of the velocity at the closest point
      end
%     compute values of velocity at required intermediate points
%     using local interpolant
      
      
      potTar = kernelDirect(vesicleUp,fup,[Xtar(J,k2);Xtar(J+Ntar,k2)],k1);
      % Need to subtract off contribution due to vesicle k1 since its
      % layer potential will be evaulted using Lagrange interpolant of
      % nearby points

      nearField(J,k2) =  nearField(J,k2) - ...
          potTar(1:numel(J));
      nearField(J+Ntar,k2) =  nearField(J+Ntar,k2) - ...
          potTar(numel(J)+1:end);
      
      XLag = zeros(2*numel(J),interpOrder - 1);
      % initialize space for initial tracer locations
      for i = 1:numel(J)
        nx = (Xtar(J(i),k2) - nearest{k1}(J(i),k2))/...
            dist{k1}(J(i),k2);
        ny = (Xtar(J(i)+Ntar,k2) - nearest{k1}(J(i)+Ntar,k2))/...
            dist{k1}(J(i),k2);
        XLag(i,:) = nearest{k1}(J(i),k2) + ...
            beta*h*nx*(1:interpOrder-1);
        XLag(i+numel(J),:) = nearest{k1}(J(i)+Ntar,k2) + ...
            beta*h*ny*(1:interpOrder-1);
        % Lagrange interpolation points coming off of vesicle k1 All
        % points are behind Xtar(J(i),k2) and are sufficiently far from
        % vesicle k1 so that the Nup-trapezoid rule gives sufficient
        % accuracy
      end

     
      lagrangePts = kernelDirect(vesicleUp,fup,XLag,k1);
      % evaluate velocity at the lagrange interpolation points
      
      for i = 1:numel(J)
        Px = o.interpMat*[vel(J(i),k2,k1) ...
            lagrangePts(i,:)]';
        Py = o.interpMat*[vel(J(i)+Ntar,k2,k1) ...
            lagrangePts(i+numel(J),:)]';
        % Build polynomial interpolant along the one-dimensional
        % points coming out of the vesicle
        dscaled = full(dist{k1}(J(i),k2)/(beta*h*(interpOrder-1)));
        % Point where interpolant needs to be evaluated

        v = filter(1,[1 -dscaled],Px);
        nearField(J(i),k2) = nearField(J(i),k2) + ...
            v(end);
        v = filter(1,[1 -dscaled],Py);
        nearField(J(i)+Ntar,k2) = nearField(J(i)+Ntar,k2) + ...
            v(end);
        % Assign higher-order results coming from Lagrange 
        % integration to velocity at near point.  Filter is faster
        % than polyval
      end % i
    end % numel(J) ~= 0
    % Evaluate layer potential at Lagrange interpolation
    % points if there are any
  end % k2
end % k1
% farField

LP = farField + nearField;

end % nearSingInt
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [qw, qp, Rbac, Rfor] = singQuadStokesSLmatrix(o,N)

% everything below must be created once, since they are constants
% the weights
v = zeros(7,1); u = zeros(7,1);

v(1,1) = 6.531815708567918e-3;
u(1,1) = 2.462194198995203e-2;

v(2,1) = 9.086744584657729e-2;
u(2,1) = 1.701315866854178e-1;

v(3,1) = 3.967966533375878e-1;
u(3,1) = 4.609256358650077e-1;

v(4,1) = 1.027856640525646e+0;
u(4,1) = 7.947291148621895e-1;

v(5,1) = 1.945288592909266e+0;
u(5,1) = 1.008710414337933e+0;

v(6,1) = 2.980147933889640e+0;
u(6,1) = 1.036093649726216e+0;

v(7,1) = 3.998861349951123e+0;
u(7,1) = 1.004787656533285e+0;
a = 5;



% get the weights coming from Table 8 of Alpert's 1999 paper

h = 2*pi/N;
n = N - 2*a + 1;

of = fft1_py;
A1 = of.sinterpS(N,v*h);
A2 = of.sinterpS(N,2*pi-flipud(v*h));
yt = h*(a:n-1+a)';
% regular points away from the singularity
wt = [h*u; h*ones(length(yt),1); h*flipud(u)]/4/pi;
% quadrature points away from singularity

B = sparse(length(yt),N);
pos = 1 + (a:n-1+a)';

for k = 1:length(yt)
  B(k, pos(k)) = 1;
end
A = [sparse(A1); B; sparse(A2)];
qw = [wt, A];

qp = qw(:,2:end);
qw = qw(:,1);

ind = (1:N)';
Rfor = zeros(N);
Rbac = zeros(N);
% vector of indicies so that we can apply circshift to each column
% efficiently.  Need one for going 'forward' and one for going
% 'backwards'
Rfor(:,1) = ind;
Rbac(:,1) = ind;
for k = 2:N
  Rfor(:,k) = (k-1)*N + [ind(k:N);ind(1:k-1)];
  Rbac(:,k) = (k-1)*N + [ind(N-k+2:N);ind(1:N-k+1)];
end

end % singQuadStokesSLmatrix

end % methods


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
methods (Static)

function LP = lagrangeInterp
% interpMap = lagrangeInterp builds the Lagrange interpolation
% matrix that takes seven function values equally distributed
% in [0,1] and returns the seven polynomial coefficients

interpMat = zeros(7);
LP(1,1) = 6.48e1;
LP(1,2) = -3.888e2;
LP(1,3) = 9.72e2;
LP(1,4) = -1.296e3;
LP(1,5) = 9.72e2;
LP(1,6) = -3.888e2;
LP(1,7) = 6.48e1;

LP(2,1) = -2.268e2;
LP(2,2) = 1.296e3;
LP(2,3) = -3.078e3;
LP(2,4) = 3.888e3;
LP(2,5) = -2.754e3;
LP(2,6) = 1.0368e3;
LP(2,7) = -1.62e2;

LP(3,1) = 3.15e2;
LP(3,2) = -1.674e3;
LP(3,3) = 3.699e3;
LP(3,4) = -4.356e3;
LP(3,5) = 2.889e3;
LP(3,6) = -1.026e3;
LP(3,7) = 1.53e2;

LP(4,1) = -2.205e2;
LP(4,2) = 1.044e3;
LP(4,3) = -2.0745e3;
LP(4,4) = 2.232e3;
LP(4,5) = -1.3815e3;
LP(4,6) = 4.68e2;
LP(4,7) = -6.75e1;

LP(5,1) = 8.12e1;
LP(5,2) = -3.132e2;
LP(5,3) = 5.265e2;
LP(5,4) = -5.08e2;
LP(5,5) = 2.97e2;
LP(5,6) = -9.72e1;
LP(5,7) = 1.37e1;

LP(6,1) = -1.47e1;
LP(6,2) = 3.6e1;
LP(6,3) = -4.5e1;
LP(6,4) = 4.0e1;
LP(6,5) = -2.25e1;
LP(6,6) = 7.2e0;
LP(6,7) = -1e0;

LP(7,1) = 1e0;
% rest of the coefficients are zero

end % lagrangeInterp
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end % methods (Static)
end % classdef