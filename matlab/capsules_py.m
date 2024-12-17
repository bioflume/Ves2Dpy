classdef capsules_py < handle
% This class implements standard calculations that need to
% be done to a vesicle, solid wall, or a collection of arbitrary
% target points (such as tracers or pressure/stress targets)
% Given a vesicle, the main tasks that can be performed are
% computing the required derivatives (bending, tension, surface
% divergence), the traction jump, the pressure and stress, 
% and constructing structures required for near-singluar
% integration

properties
N;          % number of points per vesicle
nv;         % number of vesicles
X;          % positions of vesicles
sig;        % tension of vesicles
u;          % velocity field of vesicles
kappa;      % bending modulus
viscCont;   % viscosity contrast
xt;         % tangent unit vector
sa;         % Jacobian
isa;        % inverse of Jacobian
length;     % minimum length over allc vesicles
cur;        % curvature
center;     % center of the point required for stokeslets
            % and rotlets
IK;         % index of Fourier modes for fft and ifft
            % that are needed repetatively

end %properties

methods

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function o = capsules_py(X,sigma,u,kappa,viscCont)
% capsules(X,sigma,u,kappa,viscCont) sets parameters and options for
% the class; no computation takes place here.  
%
% sigma and u are not needed and typically unknown, so just set them to
% an empty array.

o.N = size(X,1)/2;              % points per vesicle
o.nv = size(X,2);               % number of vesicles
o.X = X;                        % position of vesicle
oc = curve_py;
% Jacobian, tangent, and curvature
[o.sa,o.xt,o.cur] = oc.diffProp(o.X);
o.isa = 1./o.sa;
o.sig = sigma;          % Tension of vesicle
o.u = u;                % Velocity of vesicle
o.kappa = kappa;        % Bending modulus
o.viscCont = viscCont;  % Viscosity contrast
% center of vesicle.  Required for center of rotlets and
% stokeslets in confined flows
o.center = [mean(X(1:o.N,:));mean(X(o.N+1:2*o.N,:))];

% minimum arclength needed for near-singular integration
[~,~,len] = oc.geomProp(X);
o.length = min(len);

% ordering of the fourier modes.  It is faster to compute once here and
% pass it around to the fft differentitation routine
o.IK = fft1_py.modes(o.N,o.nv);


end % capsules
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function f = tracJump(o,f,sigma)
% tracJump(f,sigma) computes the traction jump where the derivatives
% are taken with respect to a linear combiation of previous time steps
% which is stored in object o Xm is 2*N x nv and sigma is N x nv

f = o.bendingTerm(f) + o.tensionTerm(sigma);

end % tracJump

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function ben = bendingTerm(o,f)
% ben = bendingTerm(f) computes the term due to bending
% -kappa*fourth-order derivative

ben = [-o.kappa*curve_py.arcDeriv(f(1:o.N,:),4,o.isa,o.IK);...
  -o.kappa*curve_py.arcDeriv(f(o.N+1:2*o.N,:),4,o.isa,o.IK)];

end % bendingTerm
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function repForce = repulsionForce(o,X,W)
% rep = repulsionForce(o,X,W) computes the artificial repulsion between vesicles. 
% W is the repulsion strength -- depends on the length and velocity scale
% of the flow.
% 
% Repulsion is computed using the discrete penalty layers given in Grinspun
% et al. (2009), Asynchronuous Contact Mechanics.

oc = curve;
nv = numel(X(1,:));
N = numel(X(:,1))/2;

% Compute x,y coordinates of net repulsive force on each point of each
% vesicle due to all other vesicles and walls
repForce = zeros(2*N,nv);

ox = X(1:end/2,:);
oy = X(end/2+1:end,:);

for k = 1:nv
  repx = zeros(N,1); repy = zeros(N,1);  
  
  notk = [(1:k-1) (k+1:nv)];
  notk_ox = ox(:,notk);
  notk_oy = oy(:,notk);
  
  for j = 1 : N
    if nv > 1  
      % Find the distances to each point on a vesicle  
      dist = ((ox(j,k) - notk_ox).^2 + ...
          (oy(j,k) - notk_oy).^2).^0.5;

      % Find out the maximum amount of layers necessary for each distance
      L = floor(eta./dist);

      % Stiffness l^2
      dF = -L.*(2*L+1).*(L+1)/3 + L.*(L+1).*eta./dist;
      
      repx(j) = sum(sum(dF .* (ox(j,k) - notk_ox)));
      repy(j) = sum(sum(dF .* (oy(j,k) - notk_oy)));
    end
  end % o.N
  % repulsion on the kth vesicle multiplied with strength W
  repForce(:,k) = W*[repx;repy];
end % nv
end % repulsionForce
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function ten = tensionTerm(o,sig)
% ten = tensionTerm(o,sig) computes the term due to tension (\sigma *
% x_{s})_{s}

ten = [curve_py.arcDeriv(sig.*o.xt(1:o.N,:),1,o.isa,o.IK);...
    curve_py.arcDeriv(sig.*o.xt(o.N+1:2*o.N,:),1,o.isa,o.IK)];

end % tensionTerm

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function f = surfaceDiv(o,f)
% divf = surfaceDiv(f) computes the surface divergence of f with respect
% to the vesicle stored in object o.  f has size N x nv

oc = curve_py; 
[fx,fy] = oc.getXY(f);
[tangx,tangy] = oc.getXY(o.xt);
f = curve_py.arcDeriv(fx,1,o.isa,o.IK).*tangx + ...
  curve_py.arcDeriv(fy,1,o.isa,o.IK).*tangy;

end % surfaceDiv


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Ben,Ten,Div] = computeDerivs(o)
% [Ben,Ten,Div] = computeDerivs computes the matricies that takes a
% periodic function and maps it to the fourth derivative, tension, and
% surface divergence all with respect to arclength.  Everything in this
% routine is matrix free at the expense of having repmat calls

Ben = zeros(2*o.N,2*o.N,o.nv);
Ten = zeros(2*o.N,o.N,o.nv);
Div = zeros(o.N,2*o.N,o.nv);

D1 = fft1_py.fourierDiff(o.N);

for k = 1:o.nv
  % compute single arclength derivative matrix
  isa = o.isa(:,k);
  arcDeriv = isa(:,ones(o.N,1)).*D1;
  % This line is equivalent to repmat(o.isa(:,k),1,o.N).*D1 but much
  % faster.

  D4 = arcDeriv*arcDeriv; D4 = D4*D4;
  Ben(:,:,k) = [D4 zeros(o.N); zeros(o.N) D4];

  Ten(:,:,k) = [arcDeriv*diag(o.xt(1:o.N,k));...
               arcDeriv*diag(o.xt(o.N+1:end,k))];

  Div(:,:,k) = [diag(o.xt(1:o.N,k))*arcDeriv ...
               diag(o.xt(o.N+1:end,k))*arcDeriv];
end
Ben = real(Ben);
Ten = real(Ten);
Div = real(Div);
% Imaginary part should be 0 since we are preforming a real operation

end % computeDerivs

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [NearSelf,NearOther] = getZone(vesicle1,vesicle2,relate)
  % [NearSelf,NearOther] = getZone(vesicle1,vesicle2,relate) constructs
  % each vesicle, index of the closest point, nearest point on a local
  % interapolant, and argument of that nearest point.  vesicle1 contains
  % the source points (which are also target points) and vesicle2 
  % contains additional target points.  The 
  % values of relate corresond to
  % relate == 1  => only require NearSelf  (ie. vesicle to vesicle)
  % relate == 2  => only require NearOther (ie. vesicle to wall)
  % relate == 3  => require both NearSelf and NearOther
  % THIS ROUTINE HAS A LOOP OVER THE TARGET POINTS WHICH SHOULD BE REMOVED
  NearSelf = [];
  NearOther = [];
  
  N1 = vesicle1.N; % number of source/target points
  nv1 = vesicle1.nv; % number of source/target vesicles
  X1 = vesicle1.X; % source and target points
  oc = curve_py;
  [xsou,ysou] = oc.getXY(X1); 
  % separate targets into x and y coordinates
  
  h = vesicle1.length/N1; 
  % smallest arclength over all vesicles
  ptsperbox = 10; 
  % Estimate for number of points per box.  This simply sets the 
  % number of uniformly refined boxes we take.  Estimate is not very
  % accurate.  What ptsperbox represents is the total number of points
  % that could be put in each two-dimensional bin where no two are
  % less than distance h from one another.  However, our points live
  % on curves and thus will not fill up an entire bin
  
  H = sqrt(ptsperbox)*h;
  xmin = min(min(xsou));
  xmax = max(max(xsou));
  xmin = xmin - H;
  xmax = xmax + H;
  ymin = min(min(ysou));
  ymax = max(max(ysou));
  ymin = ymin - H;
  ymax = ymax + H;
  % Add a buffer around the points so that it is easier to
  % work with vesicle2
  
  Nx = ceil((xmax - xmin)/H);
  Ny = ceil((ymax - ymin)/H);
  % Find bounds for box that contains all points and add a buffer 
  % so that all points are guaranteed to be in the box
  
  Nbins = Nx * Ny; % Total number of bins
  
  ii = ceil((xsou - xmin)/H);
  jj = ceil((ysou - ymin)/H);
  % Index in x and y direction of the box containing each point
  bin = (jj-1)*Nx + ii;
  % Find bin of each point using lexiographic ordering (x then y)
  
  
  %figure(2);
  %clf; hold on
  %plot(xsou,ysou,'k.')
  %axis equal
  %axis([xmin xmin+Nx*H ymin ymin+Ny*H])
  %set(gca,'xtick',linspace(xmin,xmin+Nx*H,Nx+1))
  %set(gca,'ytick',linspace(ymin,ymin+Ny*H,Ny+1))
  %grid on
  %set(gca,'xticklabel',[])
  %set(gca,'yticklabel',[])
  %figure(1)
  %pause
  % DEBUG: This does a simple plot of the points with a grid that 
  % aligns with the boundary of the boxes
  
  %whos
  %disp([Nbins nv1])
  %disp([xmin xmax ymin ymax])
  %disp([h H])
  %pause
  fpt = zeros(Nbins,nv1);
  lpt = zeros(Nbins,nv1);
  % allocate space for storing first and last points
  [binsort,permute] = sort(bin);
  % build permute.  Need binsort to find first and last points
  % in each box
  
  for k = 1:nv1 % Loop over vesicles
    for j = 1:N1 % Loop over bins
      ibox = binsort(j,k);
      if (fpt(ibox,k) == 0)
        fpt(ibox,k) = j;
        lpt(ibox,k) = 1;
      else
        lpt(ibox,k) = lpt(ibox,k) + 1;
      end
    end
    lpt(:,k) = fpt(:,k) + lpt(:,k) - 1;
  end
  % Construct first and last point in each box corresponding
  % to each vesicle.  The order is based on permute.  For example,
  % permute(fpt(ibox,k)),...,permute(lpt(ibox,k)) is the set of 
  % all points from vesicle k contained in box ibox
  
  neigh = zeros(Nbins,9);
  
  %Do corners first
  neigh(1,1:4) = [1 2 Nx+1 Nx+2]; 
  % bottom left corner
  neigh(Nx,1:4) = [Nx Nx-1 2*Nx 2*Nx-1]; 
  % bottom right corner
  neigh(Nbins-Nx+1,1:4) = [Nbins-Nx+1 Nbins-Nx+2 ...
      Nbins-2*Nx+1 Nbins-2*Nx+2];
  % top left corner
  neigh(Nbins,1:4) = [Nbins Nbins-1 Nbins-Nx Nbins-Nx-1]; 
  % top right corner
  
  for j = 2:Nx-1
    neigh(j,1:6) = j + [-1 0 1 Nx-1 Nx Nx+1];
  end
  % neighbors of bottom row
  
  for j = Nbins-Nx+2:Nbins-1
    neigh(j,1:6) = j + [-1 0 1 -Nx-1 -Nx -Nx+1];
  end
  % neighbors of top row
  
  for j=Nx+1:Nx:Nbins-2*Nx+1
    neigh(j,1:6) = j + [-Nx -Nx+1 0 1 Nx Nx+1];
  end
  % neighbors of left column
  
  for j=2*Nx:Nx:Nbins-Nx
    neigh(j,1:6) = j + [-Nx-1 -Nx -1 0 Nx-1 Nx];
  end
  % neighbors of right column
  
  J = (Nx + 1:Nbins - Nx);
  J = J(mod(J-1,Nx)~=0);
  J = J(mod(J,Nx)~=0);
  % J is the index of boxes that are not on the boundary
  for j=J
    neigh(j,:) = j + [-Nx-1 -Nx -Nx+1 -1 0 1 Nx-1 Nx Nx+1];
  end
  % neighbors of interior points
  % TREE STRUCTURE IS COMPLETE
  
  
  if (relate == 1 || relate == 3)
  %  distSS = -ones(N1,nv1,nv1); 
  %  % dist(n,k,j) is the distance of point n on vesicle k to
  %  % vesicle j
  %  zoneSS = -ones(N1,nv1,nv1); 
  %  % near or far zone
  %  nearestSS = -ones(2*N1,nv1,nv1); 
  %  % nearest point using local interpolant
  %  icpSS = -ones(N1,nv1,nv1); 
  %  % index of closest discretization point
  %  argnearSS = -ones(N1,nv1,nv1); 
  %  % argument in [0,1] of local interpolant
    for k = 1:nv1
      distSS{k} = spalloc(N1,nv1,0);
      % dist(n,k,j) is the distance of point n on vesicle k to
      zoneSS{k} = spalloc(N1,nv1,0);
      % near or far zone
      nearestSS{k} = spalloc(2*N1,nv1,0);
      % nearest point using local interpolant
      icpSS{k} = spalloc(N1,nv1,0);
      % index of closest discretization point
      argnearSS{k} = spalloc(N1,nv1,0);
      % argument in [0,1] of local interpolant
    end
    % New way of representing near-singular integration structure so that
    % we can use sparse matricies.
  
  
    % begin classifying points where we are considering 
    % vesicle to vesicle relationships
    for k = 1:nv1
      boxes = unique(bin(:,k));
      % Find all boxes containing points of vesicle k
      boxes = neigh(boxes,:);
      % Look at all neighbors of boxes containing vesicle k
      boxes = unique(boxes(:));
      % Remove repetition
      boxes = boxes(boxes~=0);
      % Delete non-existent boxes that came up because of neigh
  
      K = [(1:k-1) (k+1:nv1)];
      for k2 = K
        istart = fpt(boxes,k2);
        iend = lpt(boxes,k2);
        istart = istart(istart ~= 0);
        iend = iend(iend ~= -1);
        % Find index of all points in neighboring boxes of vesicle k
        % that are in vesicle k2
        
        neighpts = zeros(sum(iend-istart+1),1);
        
        % Allocate space to assign possible near points
        is = 1;
        for j=1:numel(istart)
          ie = is + iend(j) - istart(j);
          neighpts(is:ie) = permute(istart(j):iend(j),k2);
          is = ie + 1;
        end
        % neighpts contains all points on vesicle k2 that are in 
        % neighboring boxes to vesicle k
  
        neighpts = sort(neighpts);
        % sorting should help speedup as we won't be jumping around
        % through different boxes
  
        n = 0;
        for i=1:numel(neighpts)
          ipt = neighpts(i);
          ibox = bin(ipt,k2);
          % box containing ipt on vesicle k2
          if (ibox ~= n)
            n = ibox;
            % Check if we need to move to a new box
            neighbors = neigh(ibox,:);
            % neighbors of this box
            neighbors = neighbors(neighbors~=0);
            % Remove non-existent neighbors
            istart = fpt(neighbors,k);
            iend = lpt(neighbors,k);
            istart = istart(istart ~= 0);
            iend = iend(iend ~= -1);
            % Find points on vesicle k in neighboring boxes
            neighpts2 = zeros(sum(iend-istart+1),1);
            is = 1;
            for j=1:numel(istart)
              ie = is + iend(j) - istart(j);
              neighpts2(is:ie) = permute(istart(j):iend(j),k);
              is = ie + 1;
            end
            % neighpts2 contains all points on vesicle k that 
            % are in neighboring box of ibox 
          end % decide if we need to switch boxes
  
          [d0,d0loc] = min((xsou(ipt,k2) - xsou(:,k)).^2 + ...
              (ysou(ipt,k2) - ysou(:,k)).^2);
          % Find minimum distance between ipt on vesicle k2 to
          % possible closest points on vesicle k
          d0 = sqrt(d0);
          % Save on not taking the square root on a vector but instead
          % on a single real number
  
          icpSS{k}(ipt,k2) = d0loc;
          if (d0 < 2*h);
            [distSS{k}(ipt,k2),nearestx,nearesty,argnearSS{k}(ipt,k2)] = ...
                vesicle1.closestPnt([xsou;ysou],xsou(ipt,k2),...
                ysou(ipt,k2),k,icpSS{k}(ipt,k2));
            nearestSS{k}(ipt,k2) = nearestx;
            nearestSS{k}(ipt+N1,k2) = nearesty;
            % Find closest point along a local interpolant using
            % Newton's method.
  
            if (distSS{k}(ipt,k2) < h)
              zoneSS{k}(ipt,k2) = 1;
            end
            % Point ipt of vesicle k2 is in the near zone of
            % vesicle k
          end
  
  
        end % ipt
  
      end % k2
  
    end % k
  
    NearSelf.dist = distSS;
    NearSelf.zone = zoneSS;
    NearSelf.nearest = nearestSS;
    NearSelf.icp = icpSS;
    NearSelf.argnear = argnearSS;
    % Store everything in the structure NearSelf.  This way it is 
    % much cleaner to pass everything around
  end % relate == 1 || relate == 3
  
  % Bin target points with respect to the source points
  if (relate == 2 || relate == 3)
    N2 = vesicle2.N; % number of additional targets
    nv2 = vesicle2.nv; % number of additional vesicles
    X2 = vesicle2.X; % additional target points
    [xtar,ytar] = oc.getXY(X2);
    % separate additional target points into x and y coordinates
  %  figure(2); clf
  %  plot(xtar,ytar,'r.')
  %  hold on
  %  plot(xsou,ysou,'k.')
  %  pause
  % DEBUG: FOR SEEING TARGET AND SOURCE POINTS IN THE TREE STRUCTURE
  % WHICH CAN BE PLOTTED ABOVE
  
    for k = 1:nv1
      distST{k} = spalloc(N1,nv2,0);
      % dist(n,k,j) is the distance of point n on vesicle k to
      zoneST{k} = spalloc(N1,nv2,0);
      % near or far zone
      nearestST{k} = spalloc(2*N1,nv2,0);
      % nearest point using local interpolant
      icpST{k} = spalloc(N1,nv2,0);
      % index of closest discretization point
      argnearST{k} = spalloc(N1,nv2,0);
      % argument in [0,1] of local interpolant
    end
    % Represent near-singular integration structure using sparse matricies
  
    itar = ceil((xtar - xmin)/H);
    jtar = ceil((ytar - ymin)/H);
    [indx,indy] = find((itar >= 1) & (itar <= Nx) & ...
        (jtar >= 1) & (jtar <= Ny));
    % Only have to consider xx(ind),yy(ind) since all other points
    % are not contained in the box [xmin xmax] x [ymin ymax]
  
    for k = 1:nv1 % loop over sources
      for nind = 1:numel(indx) 
        % loop over points that are not outside the box that surrounds
        % all target points with a sufficiently large buffer
        ii = indx(nind);
        jj = indy(nind);
        binTar = (jtar(ii,jj)-1)*Nx + itar(ii,jj);
        boxesTar = neigh(binTar,:);
        boxesTar = boxesTar(boxesTar~=0);
        istart = fpt(boxesTar,k);
        iend  = lpt(boxesTar,k);
        istart = istart(istart ~= 0);
        iend   = iend(iend ~= -1);
        
        neighpts = zeros(sum(iend-istart+1),1);
    
        % Allocate space to assign possible near points
        if numel(neighpts) > 0
          % it is possible of the neighboring boxes to contain
          % no points.
          is = 1;
          
          for j = 1:numel(istart)
            ie = is + iend(j) - istart(j);
            neighpts(is:ie) = permute(istart(j):iend(j),k);
            is = ie + 1;
          end
          % Set of potentially nearest points to 
          % (xtar(jj),ytar(jj))
          
          [d0,d0loc] = min((xtar(ii,jj) - xsou(neighpts,k)).^2 + ...
            (ytar(ii,jj) - ysou(neighpts,k)).^2);
          % find closest point and distance between (xtar(jj),ytar(jj))
          % and vesicle k.  Only need to look at points in neighboring
          % boxes
          
          d0 = d0.^0.5;
          icpST{k}(ii,jj) = neighpts(d0loc);
  
          if d0 < 2*h
            [distST{k}(ii,jj),nearestx,nearesty,argnearST{k}(ii,jj)] = ...
              vesicle1.closestPnt([xsou;ysou],xtar(ii,jj),...
              ytar(ii,jj),k,icpST{k}(ii,jj));
            nearestST{k}(ii,jj) = nearestx;
            nearestST{k}(ii+N2,jj) = nearesty;
            
            % DEBUG: CHECK THAT NEWTON'S METHOD HAS DONE A GOOD JOB
            % CONVERGING TO THE NEAREST POINT
            % compute distance and nearest point between 
            % (xtar(ii,jj),ytar(ii,jj)) and vesicle k
            if distST{k}(ii,jj) < h
              zoneST{k}(ii,jj) = 1;
              % (xtar(ii,jj),ytar(ii,jj)) is in the near zone of vesicle k
            end
          end % d0 < 2*h
        end % numel(neighpts) > 0
  
      end % ii and jj
  
    end % k
  
    NearOther.dist = distST;
    NearOther.zone = zoneST;
    NearOther.nearest = nearestST;
    NearOther.icp = icpST;
    NearOther.argnear = argnearST;
    % store near-singluar integration requirements in structure NearOther
  
  end % relate == 2 || relate == 3
end % getZone

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [dist,nearestx,nearesty,theta] = ...
  closestPnt(o,X,xtar,ytar,k,icp)
% [dist,nearestx,nearesty,theta] = closestPnt(X,xtar,ytar,k,icp)
% computes the closest point on vesicle k to (xtar,ytar)
% using a Lagrange interpolant.  icp is the index of the closest
% point on the discrete mesh which is used as an initial guess

N = size(X,1)/2; % Number of points per vesicle
A = o.lagrangeInterp;
interpOrder = size(A,1);
% need interpolation matrix and its size

p = ceil((interpOrder+1)/2);
% Accommodate for either an even or odd number of interpolation points
pn = mod((icp-p+1:icp-p+interpOrder)' - 1,N) + 1;
% band of points around icp.  The -1,+1 combination sets index
% 0 to N as required by the code

px = A*X(pn,k); % polynomial interpolant of x-coordinate
py = A*X(pn+N,k); % polynomial interpolant of y-coordinate
Dpx = px(1:end-1).*(interpOrder-1:-1:1)';
Dpy = py(1:end-1).*(interpOrder-1:-1:1)';
D2px = Dpx(1:end-1).*(interpOrder-2:-1:1)';
D2py = Dpy(1:end-1).*(interpOrder-2:-1:1)';
% To do Newton's method, need two derivatives

theta = 1/2;
% midpoint is a good initial guess
for newton = 1:1
zx = filter(1,[1 -theta],px);
zx = zx(end);
zy = filter(1,[1 -theta],py);
zy = zy(end);
Dzx = filter(1,[1 -theta],Dpx);
Dzx = Dzx(end);
Dzy = filter(1,[1 -theta],Dpy);
Dzy = Dzy(end);
D2zx = filter(1,[1 -theta],D2px);
D2zx = D2zx(end);
D2zy = filter(1,[1 -theta],D2py);
D2zy = D2zy(end);
% Using filter is the same as polyval, but it is much
% faster when only requiring a single polyval such as here.

newtonNum = (zx-xtar)*Dzx + (zy-ytar)*Dzy;
% numerator of Newton's method
newtonDen = (zx-xtar)*D2zx + (zy-ytar)*D2zy + ...
    Dzx^2 + Dzy^2;
% denominator of Newton's method
theta = theta - newtonNum/newtonDen;
% one step of Newton's method
end
% Do a few (no more than 3) Newton iterations

nearestx = filter(1,[1,-theta],px);
nearestx = nearestx(end);
nearesty = filter(1,[1,-theta],py);
nearesty = nearesty(end);
dist = sqrt((nearestx - xtar)^2 + (nearesty - ytar)^2);
% Compute nearest point and its distance from the target point

end % closestPnt
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function LP = lagrangeInterp(o)
% interpMap = lagrangeInterp builds the Lagrange interpolation
% matrix that takes seven function values equally distributed
% in [0,1] and returns the seven polynomial coefficients
  

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

end % methods

end %capsules



