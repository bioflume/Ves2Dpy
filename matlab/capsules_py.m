classdef capsules < handle
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
function o = capsules(X,sigma,u,kappa,viscCont)
% capsules(X,sigma,u,kappa,viscCont) sets parameters and options for
% the class; no computation takes place here.  
%
% sigma and u are not needed and typically unknown, so just set them to
% an empty array.

o.N = size(X,1)/2;              % points per vesicle
o.nv = size(X,2);               % number of vesicles
o.X = X;                        % position of vesicle
oc = curve;
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
o.IK = fft1.modes(o.N,o.nv);


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

ben = [-o.kappa*curve.arcDeriv(f(1:o.N,:),4,o.isa,o.IK);...
  -o.kappa*curve.arcDeriv(f(o.N+1:2*o.N,:),4,o.isa,o.IK)];

end % bendingTerm
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function ten = tensionTerm(o,sig)
% ten = tensionTerm(o,sig) computes the term due to tension (\sigma *
% x_{s})_{s}

ten = [curve.arcDeriv(sig.*o.xt(1:o.N,:),1,o.isa,o.IK);...
    curve.arcDeriv(sig.*o.xt(o.N+1:2*o.N,:),1,o.isa,o.IK)];

end % tensionTerm

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function f = surfaceDiv(o,f)
% divf = surfaceDiv(f) computes the surface divergence of f with respect
% to the vesicle stored in object o.  f has size N x nv

oc = curve; 
[fx,fy] = oc.getXY(f);
[tangx,tangy] = oc.getXY(o.xt);
f = curve.arcDeriv(fx,1,o.isa,o.IK).*tangx + ...
  curve.arcDeriv(fy,1,o.isa,o.IK).*tangy;

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

end % methods

end %capsules



