classdef fft1_py < handle
% class implements fft transformations.  This includes computing
% the fourier differentiation matrix, doing interplation required
% by Alpert's quadrature rules, and defining the Fourier frequencies

properties
N; % Number of points in the incoming periodic functions
end


methods

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function fo = arbInterp(o,f,y)
% fo = arbInterp(f)  interpolates the function f given at 
% regular points, at arbitrary points y.  The points y are assumed 
% to be the in the 0-2*pi range.  Matrix is built.  This routine is
% only for testing and is never called in the vesicle code

N = size(f,1);

% build interpolation matrix
A = zeros(length(y), N);
for j = 1:N
  g = zeros(N,1); g(j) =1;
  fhat = fftshift(fft(g)/N);    
  for k=1:N
    A(:, j) = A(:, j) + fhat(k)*exp(1i*(-N/2+k-1)*y);
  end
end
A = real(A);

% interpolate
fo = A*f;

end % arbInterp


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function A = sinterpS(o,N,y)
% A = sinterpS(N,y) constructs the interpolation matrix A that maps
% a function defined periodically at N equispaced points to the
% function value at the points y
% The points y are assumed to be the in the 0-2*pi range. 

A = zeros(numel(y),N);
modes = [(0:N/2-1) 0 (-N/2+1:-1)];
f = zeros(1,N);

for j=1:N
  f(j) = 1;
  fhat = fft(f)/N;
%   for k=1:N
%     A(:,j) = A(:,j) + fhat(k)*exp(1i*y*modes(k));
%   end
  fhat = fhat(ones(numel(y),1),:);
  A(:,j) = A(:,j) + sum(fhat.*exp(1i*y*modes),2);
  f(j) = 0;
end
A = real(A);
% Input is real, so interpolant should be real


end % sinterpS


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function test(o)
% test : tests the differentiation routines of fft1
% Source: Spectral Methods in Matlab, Nick Trefethen.

  fprintf('testing differentiation test:\n')
% Differentiation of a hat function:
  N = 24; h = 2*pi/N; x = h*(1:N)';
  v1 = max(0,1-abs(x-pi)/2);
  w = o.diffFT(v1); clf
  subplot(3,2,1), plot(x,v1,'.-','markersize',13)
  axis([0 2*pi -.5 1.5]), grid on, title('function')
  subplot(3,2,2), plot(x,w,'.-','markersize',13)
  axis([0 2*pi -1 1]), grid on, title('spectral derivative')

  
% Differentiation of exp(sin(x)):
  v2 = exp(sin(x)); vprime = cos(x).*v2;
  w = o.diffFT(v2);
  error = norm(w-vprime,inf);
  subplot(3,2,3), plot(x,v2,'.-','markersize',13)
  axis([0 2*pi 0 3]), grid on
  subplot(3,2,4), plot(x,w,'.-','markersize',13)
  axis([0 2*pi -2 2]), grid on
  text(2.2,1.4,['max error = ' num2str(error)])
  
  fprintf('-----------------------\n');
  fprintf('exp(sin(x)) derivative:\n');
  fprintf('-----------------------\n');
  for N=[8 16 32 64 128]
    h = 2*pi/N; x = h*(1:N)';
    v = exp(sin(x)); vprime = cos(x).*v;
    w = o.diffFT(v);
    error = norm(w-vprime,inf);
    fprintf('  %3.0f points: |e|_inf = %2.4e\n',N, error);
  end

% check interpolation
  y = rand(20,1)*2*pi;
  v2y = exp(sin(y));
  fy = o.arbInterp(v,y);
  
  fprintf('--------------------------------------------------------\n');
  fprintf('cos(x)sin(x)+sin(x)cos(10x)+sin(20x)cos(13x) first derivative:\n');
  fprintf('--------------------------------------------------------\n');
  for J=4:1:10
    N=2^J;
    h = 2*pi/N; x = h*(1:N)';
    v = cos(x).*sin(x)+sin(x).*cos(10*x)+sin(20*x).*cos(13*x);
    vprime = cos(2*x)-9/2*cos(9*x)+11/2*cos(11*x)+7/2*cos(7*x)+33/2*cos(33*x);
    w = o.diffFT(v);
    error = norm(w-vprime,inf);
    fprintf('  %3.0f points: |e|_inf = %2.4e\n',N, error);
  end

  end % test_diffFT

end % methods


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
methods (Static)

function Deriv = D1(N)
% Deriv = D1(N) constructs a N by N fourier differentiation matrix
[FF,FFI] = fft1_py.fourierInt(N);
Deriv = FFI * diag(1i*([0 -N/2+1:N/2-1])) * FF;
Deriv = real(Deriv);

end % D1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function df = diffFT(f,IK)
% df = diffFT(f,IK) Computes the first derivative of an array of 
% periodic functions f using fourier transform. The f(:,1) is the 
% first function, f(:,2) is the second function, etc.
% IK is used to speed up the code.  It is the index of the fourier
% modes so that fft and ifft can be used
% 
% EXAMPLE: 
%  N = 24; h = 2*pi/N; x = h*(1:N)';
%  IK = 1i*[(0:N/2-1) 0 (-N/2+1:-1)]';
%  v = exp(sin(x)); vprime = cos(x).*v;
%  subplot(2,2,1),plot(x,v,'.-','markersize',13);
%  axis([0 2*pi 0 3]);
%
%  w = fft1.diffFT(v,IK);
%  error = norm(w-vprime,inf);
%  subplot(2,2,2), plot(x,w,'.-','markersize',13);

df = real(ifft(IK.*fft(f)));

end % end diffFT


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function IK = modes(N,nv)
% IK = modes(N) builds the order of the fourier modes required for using
% fft and ifft to do spectral differentiation

IK = 1i*[(0:N/2-1) 0 (-N/2+1:-1)]';
% diagonal term for Fourier differentiation with the -N/2 mode
% zeroed to avoid Nyquist frequency

if nv == 2
  IK = [IK IK];
elseif nv == 3
  IK = [IK IK IK];
elseif nv > 3
  IK = repmat(IK,1,nv);
end

end % modes


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function D1 = fourierDiff(N)
% D1 = fourierDiff(N) creates the fourier differentiation matrix

D1 = fft1_py.fourierInt(N);
D1 = D1'*diag(1i*N*(-N/2:N/2-1))*D1;


end % fourierDiff

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [R,P] = fourierRandP(N,Nup)
% [R,P] = fourierRandP(N,Nup) computes the Fourier restriction and
% prolongation operators so that functions can be interpolated from N
% points to Nup points (prolongation) and from Nup points to N points
% (restriction)


R = zeros(N,Nup);
P = zeros(Nup,N);

[FF1,FFI1] = fft1_py.fourierInt(N);
[FF2,FFI2] = fft1_py.fourierInt(Nup);

R = FFI1 * [zeros(N,(Nup-N)/2) eye(N,N) zeros(N,(Nup-N)/2)] * FF2;
R = real(R);
P = R'*Nup/N;
% prolongation is transpose of restriction (with appropriate scaling)

end % fourierRandP


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [FF,FFI] = fourierInt(N)
% [FF,FFI] = fourierInt(N) returns a matrix that take in the point
% values of a function in [0,2*pi) and returns the fourier coefficients
% % (FF) and a matrix that takes in the fourier coefficients and returns
% the function values (FFI)

theta = (0:N-1)'*2*pi/N;
%modes = [0;(-N/2+1:N/2-1)'];
modes = [-N/2;(-N/2+1:N/2-1)'];

FF = exp(-1i*modes*theta')/N;
% FF takes function values and returns the Fourier coefficients

if (nargout > 1)
%   FFI = zeros(N);
%   for i=1:N
%     FFI(:,i) = exp(1i*modes(i)*theta);
%   end
  
  FFI = exp(1i*theta*modes');
  
  % FFI takes the Fourier coefficients and returns function values.
else
  FFI = [];
end


end % fourierInt


end % methods 

end % classdef
