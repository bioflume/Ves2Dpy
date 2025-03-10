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
  
  Prolong_LP
  Restrict_LP
  % prolongation and restriction matrices for layer potentitals

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
end % classdef