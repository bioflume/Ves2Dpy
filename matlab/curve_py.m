classdef curve_py
% This class implements that basic calculus on the curve.
% The basic data structure is a matrix X in which the columns 
% represent periodic C^{\infty} closed curves with N points, 
% X(1:n,j) is the x-coordinate of the j_th curve and X(n+1:N,j) 
% is the y-coordinate of the j_th curve; here n=N/2
% X coordinates do not have to be periodic, but the curvature,
% normals, etc that they compute will be garbage.  This allows
% us to store target points for tracers or the pressure using
% this class and then using near-singular integration is easy
% to implement

methods

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [x,y]=getXY(o,X) 
% GK: MAY NOT BE NECESSARY, BUT IT SHOWS HOW THE COORDINATES ARE STORED
% [x,y] = getXY(X) get the [x,y] component of curves X
N = size(X,1)/2;
x = X(1:N,:);
y = X(N+1:end,:);

end % getXY

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function V = setXY(o,x,y)
% GK: MAY NOT BE NECESSARY
% V = setXY(x,y) set the [x,y] component of vector V on the curve
N = size(x,1);
V=zeros(2*N,size(x,2));
V(1:N,:) = x;
V(N+1:end,:) = y;

end % setXY

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function center = getCenter(o,X)
% center = getCenter(o,X) finds the center of each capsule

nv = size(X,2); % number of vesicles
center = zeros(nv,1);

for k = 1 : nv
  center(k) = sqrt(mean(X(1:end/2,k))^2 + mean(X(end/2+1:end,k))^2);
end

end % getCenter

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function V = getPrincAxesGivenCentroid(o,X,center)
N = numel(X(:,1))/2;
nv = numel(X(1,:));
% compute inclination angle on an upsampled grid
for k = 1 : nv
  Xcent = [X(1:end/2,k)-center(1,k); X(end/2+1:end,k)-center(2,k)];
  
  xCent = Xcent(1:end/2,k); yCent = Xcent(end/2+1:end,k);
  [jacCent,tanCent,curvCent] = o.diffProp(Xcent);
  
  nxCent = tanCent(end/2+1:end); nyCent = -tanCent(1:end/2);
  rdotn = xCent.*nxCent + yCent.*nyCent;
  rho2 = xCent.^2 + yCent.^2;

  J11 = 0.25*sum(rdotn.*(rho2 - xCent.*xCent).*jacCent)*2*pi/N;
  J12 = 0.25*sum(rdotn.*(-xCent.*yCent).*jacCent)*2*pi/N;
  J21 = 0.25*sum(rdotn.*(-yCent.*xCent).*jacCent)*2*pi/N;
  J22 = 0.25*sum(rdotn.*(rho2 - yCent.*yCent).*jacCent)*2*pi/N;

  J = [J11 J12; J21 J22];
  [V,D] = eig(J);

  [~,ind] = min(abs(diag(D)));
  V = V(:,ind);
end

end % getPrincAxesGivenCentroid
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function center = getPhysicalCenterShan(o,X)
% center = getCenter(o,X) finds the center of each capsule
N = size(X,1)/2;
nv = size(X,2);

[jac,tan,curv] = o.diffProp(X);
tanx = tan(1:end/2,:); tany = tan(end/2+1:end,:);
nx = tany; ny = -tanx;
x = X(1:end/2,:); y = X(end/2+1:end,:);



center = zeros(2,nv);

for k = 1 : nv
    xv = (x(:,k));
    yv = (y(:,k));
    xdotn = xv.*nx(:,k); ydotn = yv.*ny(:,k);
    xdotn_sum = sum(xdotn.*jac(:,k));
    ydotn_sum = sum(ydotn.*jac(:,k));

    center(1,k) = 0.5*sum(xv.*xdotn.*jac(:,k))./xdotn_sum;
    center(2,k) = 0.5*sum(yv.*ydotn.*jac(:,k))./ydotn_sum;
end

end % getPhysicalCenterShan

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function IA = getIncAngle2(o,X)
% GK: THIS IS NEEDED IN STANDARDIZING VESICLE SHAPES 
% WE NEED TO KNOW THE INCLINATION ANGLE AND ROTATE THE VESICLE TO pi/2
% IA = getIncAngle(o,X) finds the inclination angle of each capsule
% The inclination angle (IA) is the angle between the x-axis and the 
% principal axis corresponding to the smallest principal moment of inertia
nv = size(X,2);
IA = zeros(nv,1);

% compute inclination angle on an upsampled grid
N = size(X,1)/2;
modes = [(0:N/2-1)';0;(-N/2+1:-1)'];

centX = o.getPhysicalCenterShan(X);
for k = 1 : nv
  X(:,k) = [X(1:end/2,k)-centX(1);...
      X(end/2+1:end,k)-centX(2)];
end


for k = 1 : nv
    x = X(1:end/2,k); 
    y = X(end/2+1:end,k);
    
    Dx = real(ifft(1i*modes.*fft(x)));
    Dy = real(ifft(1i*modes.*fft(y)));
    jac = sqrt(Dx.^2 + Dy.^2);
    tx = Dx./jac; ty = Dy./jac;
    nx = ty; ny = -tx; % Shan: n is the right hand side of t
    rdotn = x.*nx + y.*ny;
    rho2 = x.^2 + y.^2;

    J11 = 0.25*sum(rdotn.*(rho2 - x.*x).*jac)*2*pi/N;
    J12 = 0.25*sum(rdotn.*(-x.*y).*jac)*2*pi/N;
    J21 = 0.25*sum(rdotn.*(-y.*x).*jac)*2*pi/N;
    J22 = 0.25*sum(rdotn.*(rho2 - y.*y).*jac)*2*pi/N;

    J = [J11 J12; J21 J22];
    [V,D] = eig(J); % V are the eigenvectors, D are the eigenvalues
    
    [~,ind] = min(abs(diag(D)));
    % make sure that the first components of e-vectors have the same sign
    if V(2,ind)<0
      V(:,ind) = -1*V(:,ind);
    end
    % since V(2,ind) > 0, this will give angle between [0, pi]
    IA(k) = atan2(V(2,ind),V(1,ind));
    
    % FIND DIRECTION OF PRIN. AXIS DEPENDING ON HEAD-TAIL
    % 1) already translated to (0,0), so rotate to pi/2
    x0rot = x*cos(-IA(k)+pi/2) - y*sin(-IA(k)+pi/2);
    y0rot = x*sin(-IA(k)+pi/2) + y*cos(-IA(k)+pi/2);
    
    % 2) find areas (top, bottom)
    % need derivatives, so rotate the computed ones
    Dx = Dx*cos(-IA(k)+pi/2) - Dy*sin(-IA(k)+pi/2);
    Dy = Dx*sin(-IA(k)+pi/2) + Dy*cos(-IA(k)+pi/2);
    
    idcsTop = find(y0rot>=0); idcsBot = find(y0rot<0);
    areaTop = sum(x0rot(idcsTop).*Dy(idcsTop)-y0rot(idcsTop).*...
        Dx(idcsTop))/N*pi;
    
    areaBot = sum(x0rot(idcsBot).*Dy(idcsBot)-y0rot(idcsBot).*...
        Dx(idcsBot))/N*pi;
    
    if areaBot >= 1.1*areaTop
      IA(k) = IA(k) + pi;
    elseif areaTop < 1.1*areaBot  
      % if areaTop ~ areaBot, then check areaRight, areaLeft  
      idcsLeft = find(x0rot<0); idcsRight = find(x0rot>=0);
      areaRight = sum(x0rot(idcsRight).*Dy(idcsRight)-y0rot(idcsRight).*...
        Dx(idcsRight))/N*pi;
      areaLeft = sum(x0rot(idcsLeft).*Dy(idcsLeft)-y0rot(idcsLeft).*...
        Dx(idcsLeft))/N*pi;
      if areaLeft >= 1.1*areaRight
        IA(k) = IA(k) + pi;
      end
      % debug
    end
end
end % getIncAngle2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Dx,Dy]=getDXY(o,X)
% [Dx,Dy]=getDXY(X), compute the derivatives of each component of X 
% these are the derivatives with respect the parameterization 
% not arclength
x = X(1:end/2,:);
y = X(end/2+1:end,:);
N = size(x,1);
nv = size(x,2);
IK = fft1_py.modes(N,nv);
Dx = fft1_py.diffFT(x,IK);
Dy = fft1_py.diffFT(y,IK);

end % getDXY

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Xfinal = upsThenFilterShape(o,X,Nup,modeCut)
% delete high frequencies from the vesicle shape
N = size(X,1)/2;
nv = size(X,2);

% modeCut = 32; works fine; Nup = 512;

modes = [(0:Nup/2-1) (-Nup/2:-1)];
xup = interpft(X(1:end/2,:),Nup);
yup = interpft(X(end/2+1:end,:),Nup);

Xfinal = zeros(size(X));

for k = 1:nv
  z = xup(:,k) + 1i*yup(:,k);
  z = fft(z);
  z(abs(modes) > modeCut) = 0;
  z = ifft(z);
  Xfinal(1:end/2,k) = interpft(real(z),N);
  Xfinal(end/2+1:end,k) = interpft(imag(z),N);
end

end % filterShape
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [jacobian,tangent,curvature] = diffProp(o,X)
% [jacobian,tangent,curvature] = diffProp(X) returns differential
% properties of the curve each column of the matrix X. Each column of 
% X should be a closed curve defined in plane. The tangent is the 
% normalized tangent.
%
% EXAMPLE:
%    n = 128; nv = 3;
%    X = boundary(n,'nv',nv,'curly');
%    c = curve;
%    [k t s] = c.diffProp(X);

N = size(X,1)/2;
nv = size(X,2);

% get the x y components
[Dx,Dy] = o.getDXY(X);

jacobian = sqrt(Dx.^2 + Dy.^2); 

if nargout>1  % if user requires tangent
  tangent = o.setXY( Dx./jacobian, Dy./jacobian);
end

if nargout>2  % if user requires curvature
  IK = fft1_py.modes(N,nv);
  DDx = curve_py.arcDeriv(Dx,1,ones(N,nv),IK);
  DDy = curve_py.arcDeriv(Dy,1,ones(N,nv),IK);
  curvature = (Dx.*DDy - Dy.*DDx)./(jacobian.^3);
end
% curvature of the curve

end % diffProp

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [reducedArea,area,length] = geomProp(o,X)
% [reducedArea area length] = geomProp(X) calculate the length, area 
% and the reduced volume of domains inclose by columns of X. 
% Reduced volume is defined as 4*pi*A/L. 
% EXAMPLE:
%   X = boundary(64,'nv',3,'curly');
%   c = curve(X);
%   [rv A L] = c.geomProp(X);

[x,y] = o.getXY(X);
N = size(x,1);
[Dx,Dy] = o.getDXY(X);
length = sum(sqrt(Dx.^2 + Dy.^2))*2*pi/N;
area = sum(x.*Dy - y.*Dx)*pi/N;

reducedArea = 4*pi*area./length.^2;

end % geomProp

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function X0 = ellipse(o,N,ra)
% X0 = o.ellipse(N,ra) finds the ellipse (a*cos(theta),sin(theta)) so
% that the reduced area is ra.  Uses N points.  Parameter a is found 
% by using bisection method

t = (0:N-1)'*2*pi/N;
a = (1 - sqrt(1-ra^2))/ra;
% initial guess using approximation length = sqrt(2)*pi*sqrt(a^2+1)

X0 = [a*cos(t);sin(t)];

[raNew,~,~] = o.geomProp(X0);

cond = abs(raNew - ra)/ra < 1e-4;
maxiter = 10;
iter = 0;
while (~cond && iter < maxiter);
  iter = iter + 1;
  if (raNew > ra)
    a = 0.9 * a;
  else
    a = 1.05 * a;
  end
  % update the major axis
  X0 = [cos(t);a*sin(t)];
  % Compute new possible configuration
  [raNew,~,~] = o.geomProp(X0);
  % compute new reduced area
  cond = abs(raNew - ra) < 1e-2;
  % check if the residual is small enough
end
% iteration quits if reduced area is achieved within 1% or 
% maxiter iterations have been performed


end % ellipse

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Xnew] = correctAreaAndLength(o,X,area0,length0)
% Xnew = correctAreaAndLength(X,a0,l0) changes the shape of the vesicle
% by finding the shape Xnew that is closest to X in the L2 sense and
% has the same area and length as the original shape

% tolConstraint (which controls area and length) comes from the area-length
% tolerance for time adaptivity.

% Find the current area and length
[~,at,lt] = o.geomProp(X);
eAt = abs(at-area0)./area0;
eLt = abs(lt-length0)./length0;

N  = size(X,1)/2;
  
tolConstraint = 1e-2; % 1 percent error in constraints
% tolConstraint = timeTolerance;
tolFunctional = 1e-2; % Allowed to change shape by 1 percent

options = optimset('Algorithm','sqp','TolCon',tolConstraint,...
    'TolFun',tolFunctional,'display','off','MaxFunEvals',3000);

% Options for Algorithm are:
% 'active-set', 'interior-point', 'interior-point-convex' 'sqp'

Xnew = zeros(size(X));

for k = 1:size(Xnew,2)
    
  minFun = @(z) 1/N*min(sum((z - X(:,k)).^2));
  [Xnew(:,k),~,iflag,output] = fmincon(minFun,X(:,k),[],[],[],[],[],[],...
      @(z) o.nonlcon(z,area0(k),length0(k)),options);
  if iflag~=1 && iflag~=2
    message = ['Correction scheme failed, do not correct at this step'];
    disp(message)
    Xnew(:,k) = X(:,k);
  end
  % if fmincon fails, keep the current iterate for this time step.
  % Hopefully it'll be corrected at a later step.
  
end
disp(['Correction took ' num2str(output.iterations) ' iterations.'])
% Looping over vesicles, correct the area and length of each vesicle

end % correctAreaAndLength
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [cIn,cEx] = nonlcon(o,X,a0,l0)
% [cIn,cEx] = nonlcon(X,a0,l0) is the non-linear constraints required
% by fmincon

[~,a,l] = o.geomProp(X);

cIn = [];
% new inequalities in the constraint
cEx = [(a-a0)/a0 (l-l0)/l0];
% want to keep the area and length the same

end % nonlcon

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Xnew] = alignCenterAngle(o,Xorg,X)
% Xnew = alignCenterAngle(o,Xorg,X) uses
% rigid body translation and rotation to match X having the corrected area 
% and length but wrong center and inclination angle with Xorg having the 
% right center and IA but wrong area and length. So that, Xnew has the
% correct area,length,center and inclination angle.
N = size(X,1)/2;
nv = size(X,2);
Xnew = zeros(size(X));

for k = 1 : nv
initMean = o.getPhysicalCenterShan(Xorg); %[mean(Xorg(1:end/2,k)); mean(Xorg(end/2+1:end,k))];
newMean = o.getPhysicalCenterShan(X); %[mean(X(1:end/2,k)); mean(X(end/2+1:end,k))];

initAngle = o.getIncAngle2(Xorg(:,k));
newAngle = o.getIncAngle2(X(:,k));
if newAngle > pi
  newAngle2 = newAngle-pi;
else
  newAngle2 = newAngle+pi;
end
newAngles = [newAngle;newAngle2];
diffAngles = abs(initAngle-newAngles); [~,id] = min(diffAngles);
newAngle = newAngles(id);

% move to (0,0) new shape
Xp = [X(1:end/2,k)-newMean(1); X(end/2+1:end,k)-newMean(2)];

% tilt it to the original angle
XpNew = zeros(size(Xp)); thet = -newAngle+initAngle;
XpNew(1:end/2) = Xp(1:end/2)*cos(thet)-Xp(end/2+1:end)*sin(thet);
XpNew(end/2+1:end) = Xp(1:end/2)*sin(thet)+Xp(end/2+1:end)*cos(thet);

% move to original center
Xnew(:,k) = [XpNew(1:end/2)+initMean(1); XpNew(end/2+1:end)+initMean(2)];
end

end % alignCenterAngle

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [X,u,sigma] = redistributeArcLength(o,X,u,sigma)
% [X,u,sigma] = resdistributeArcLength(o,X,u,sigma) resdistributes
% the vesicle shape eqiuspaced in arclength and adjusts the tension and
% velocity according to the new parameterization

N = size(X,1)/2;
nv = size(X,2);
modes = [(0:N/2-1) (-N/2:-1)];
jac = o.diffProp(X);
jac1 = jac;
tol = 1e-10;

u = [];
sigma = [];


for k = 1:nv
  if norm(jac(:,k) - mean(jac(:,k)),inf) > tol*mean(jac(:,k))
    theta = o.arcLengthParameter(X(1:end/2,k),...
        X(end/2+1:end,k));
    zX = X(1:end/2,k) + 1i*X(end/2+1:end,k);
    zXh = fft(zX)/N;
    zX = zeros(N,1);
    for j = 1:N
      zX = zX + zXh(j)*exp(1i*modes(j)*theta);
    end
    X(:,k) = o.setXY(real(zX),imag(zX));
    % if nargin > 2
    %   zu = u(1:end/2,k) + 1i*u(end/2+1:end,k);
    %   zuh = fft(zu)/N;
    %   sigmah = fft(sigma(:,k))/N;
    %   zu = zeros(N,1);
    %   sigma(:,k) = zeros(N,1);
    %   for j = 1:N
    %     zu = zu + zuh(j)*exp(1i*modes(j)*theta);
    %     sigma(:,k) = sigma(:,k) + sigmah(j)*exp(1i*modes(j)*theta);
    %   end
    %   sigma = real(sigma);
    %   u(:,k) = o.setXY(real(zu),imag(zu));
    % else
    %   u = [];
    %   sigma = [];
    % end
    % redistribute the vesicle positions and tension so that it is
    % equispaced in arclength
  end
end

end % redistributeArcLength

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [theta,arcLength] = arcLengthParameter(o,x,y)
% theta = arcLengthParamter(o,x,y) finds a discretization of parameter
% space theta so that the resulting geometry will be equispaced in
% arclength

N = numel(x);
t = (0:N-1)'*2*pi/N; 
[~,~,len] = o.geomProp([x;y]);
% find total perimeter
[Dx,Dy] = o.getDXY([x;y]);
% find derivative
arc = sqrt(Dx.^2 + Dy.^2);
arch = fft(arc);
modes = -1i./[(0:N/2-1) 0 (-N/2+1:-1)]';
modes(1) = 0;
modes(N/2+1) = 0;
arcLength = real(ifft(modes.*arch) - sum(modes.*arch/N) + arch(1)*t/N);

z1 = [arcLength(end-6:end)-len;arcLength;arcLength(1:7)+len];
z2 = [t(end-6:end)-2*pi;t;t(1:7)+2*pi];
% put in some overlap to account for periodicity

theta = [interp1(z1,z2,(0:N-1)'*len/N,'spline')];

end % arcLengthParamter


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [X,niter] = reparametrize(o,X,dX,maxIter)
% [X,niter] = reparametrize applies the reparametrization with
% minimizing the energy in the high frequencies (Veerapaneni et al. 2011, 
% doi: 10.1016/j.jcp.2011.03.045, Section 6).

% Decay power (k^pow)
pow = 4;

N     = size(X,1)/2; % # of points per vesicle
nv    = size(X,2)  ; % # of vesicles

niter = ones(nv,1); % store # of iterations per vesicle
tolg = 1e-3;
if isempty(dX)
  [~,~,len] = o.geomProp(X);    
  dX = len/N;
  toly = 1e-5*dX;
else
  normDx = sqrt(dX(1:end/2,:).^2+dX(end/2+1:end,:).^2);
  toly = 1e-3*min(normDx(:));  
end


beta = 0.1;
dtauOld = 0.05;

for k = 1:nv
    
    % Get initial coordinates of kth vesicle (upsample if necessary)
    x0 = X(1:end/2,k); 
    y0 = X(end/2+1:end,k);
    
    % Compute initial projected gradient energy
    g0 = o.computeProjectedGradEnergy(x0,y0,pow);  
    x = x0; y = y0; g = g0;
    
    % Explicit reparametrization
    while niter(k) <= maxIter
        dtau = dtauOld;
        xn = x - g(1:end/2)*dtau; yn = y - g(end/2+1:end)*dtau;
        gn = o.computeProjectedGradEnergy(xn,yn,pow);
        while norm(gn) > norm(g)
            dtau = dtau*beta;
            xn = x - g(1:end/2)*dtau; yn = y - g(end/2+1:end)*dtau;
            gn = o.computeProjectedGradEnergy(xn,yn,pow);
        end
        dtauOld = dtau*1/beta;
        
        if norm(gn) < max(toly/dtau,tolg*norm(g0))
            break
        end
        
        x = xn; y = yn; g = gn;
        niter(k) = niter(k)+1;
    end
    X(:,k) = [xn;yn];
end

end % end reparamEnergyMin

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [g] = computeProjectedGradEnergy(o,x,y,pow)
% g = computeProjectedGradEnergy(o,x,y) computes the projected gradient of
% the energy of the surface. We use this in reparamEnergyMin(o,X). For the
% formulation see (Veerapaneni et al. 2011 doi: 10.1016/j.jcp.2011.03.045,
% Section 6)

N = numel(x);

% to be used in computing gradE 
modes = [(0:N/2-1) (-N/2:-1)]'; 

% get tangent vector at each point (tang_x;tang_y) 
[~,tang] = o.diffProp([x;y]);
% get x and y components of normal vector at each point
nx = tang(N+1:2*N);
ny = -tang(1:N);

% Compute gradE
% first, get Fourier coefficients
zX = x + 1i*y;
zXh = fft(zX)/N;
% second, compute zX with a_k = k^pow
zX = ifft(N*zXh.*abs(modes).^pow);

% Compute Energy
gradE = [real(zX);imag(zX)]; % [gradE_x;gradE_y]

% A dyadic product property (a (ban) a)b = a(a.b) can be used to avoid the
% for loop as follows
normals = [nx;ny];
% do the dot product n.gradE
prod = normals.*gradE;
dotProd = prod(1:N)+prod(N+1:2*N);
% do (I-(n ban n))gradE = gradE - n(n.gradE) for each point
g = gradE - normals.*[dotProd;dotProd];
end % end computeProjectedGradEnergy

end % methods
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
methods (Static)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function f = arcDeriv(f,m,isa,IK)
% f = arcDeriv(f,m,s,IK,col) is the arclength derivative of order m.
% f is a matrix of scalar functions (each function is a column)
% f is assumed to have an arbitrary parametrization
% isa = d a/ d s, where a is the aribtrary parameterization
% IK is the fourier modes which is saved and used to accelerate 
% this routine

for j=1:m
  f = isa.*ifft(IK.*fft(f));
end
f = real(f);

end % arcDeriv
end % methods (Static)

end % classdef
