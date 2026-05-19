clear; clc;

% load curve_py
oc = curve_py;

% file name
fileName = ['./output/test.bin']; % to save simulation data

% Flow specification
bgFlow = 'parabolic';
speed = 12000; 
vinf = setBgFlow(bgFlow, speed);


% time stepping
dt = 1e-5; % time step size
Th = 0.15; % time horizon

% vesicle discretization
N = 128; % number of points to discretize vesicle (default 128 as networks)

% vesicle initialization
nv = 1;  % number of vesicles
ra = 0.65; % reuced area -- networks trained with vesicles of ra = 0.65
X0 = oc.ellipse(N,ra); % initial vesicle as ellipsoid

% arc-length is supposed to be 1 so divide by the initial length
[~,area0,len0] = oc.geomProp(X0);
X0 = X0/len;

center = [0; 0.065]; % center in [x, y]
IA = pi/2; % inclination angle
X = zeros(size(X0));
X(1:N) = cos(IA) * X0(1:N) - sin(IA) * X0(N+1:2*N) + center(1);
X(N+1:2*N) = sin(IA) * X0(1:N) +  cos(IA) * X0(N+1:2*N) + center(2);

% build mlarm class to take time steps using networks
% load the normalization (mean, std) values for the networks
load ./shannets/ves_fft_in_param.mat % loads advNetInputNorm
load ./shannets/ves_fft_out_param.mat  % loads advNetOutputNorm
load ./shannets/ves_relax_dt1E5.mat % loads relaxNetInputNorm, relaxNetOutputNorm

mlarm = MLARM_py(dt,vinf,oc,advNetInputNorm,advNetOutputNorm,...
    relaxNetInputNorm,relaxNetOutputNorm);
[~,mlarm.area0,mlarm.len0] = oc.geomProp(X);

% save the initial data 
fid = fopen(fileName,'w');
fwrite(fid, [N;nv;X(:)], 'double');
fclose(fid);

% evolve in time
time = 0; it = 0;
while time < Th

  % take a time step
  tStart = tic;
  X = mlarm.time_step(X); 
  tEnd = toc;

  % find error in area and length
  [~,area,len] = oc.geomProp(X);
  errArea = max(abs(area-mlarm.area0)./mlarm.area0); 
  errLen = max(abs(len-mlarm.len0)./mlarm.len0);

  % update counter and time
  it = it + 1;
  time = time + dt;

  % print time step info
  disp('********************************************') 
  disp([num2str(it) 'th time step, time: ' num2str(time)])
  disp(['Solving with networks takes ' num2str(tEnd-tStart) ' sec.'])
  disp(['Error in area and length: ' num2str(max(errArea, errLen))])   
  disp('********************************************') 
  disp(' ')

  % save data
  output = [time;X(:)];
  fid = fopen(fileName,'a');
  fwrite(fid,output,'double');
  fclose(fid);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function vinf = setBgFlow(bgFlow,speed)    
  vinf  = @(X) zeros(size(X));      
  if strcmp(bgFlow,'relax')
    vinf = @(X) zeros(size(X));  % relaxation
  elseif strcmp(bgFlow,'shear') 
    vinf = @(X) speed*[X(end/2+1:end,:);zeros(size(X(1:end/2,:)))]; 
  elseif strcmp(bgFlow,'tayGreen')
    vinf = @(X) speed*[sin(X(1:end/2,:)).*cos(X(end/2+1:end,:));-...
      cos(X(1:end/2,:)).*sin(X(end/2+1:end,:))]; % Taylor-Green
  elseif strcmp(bgFlow,'parabolic')
    vinf = @(X) [speed*(1-(X(end/2+1:end,:)/1.3).^2);...
        zeros(size(X(1:end/2,:)))];
  elseif strcmp(bgFlow,'rotation')
    vinf = @(X) [-sin(atan2(X(end/2+1:end,:),X(1:end/2,:)))./sqrt(X(1:end/2,:).^2+X(end/2+1:end,:).^2);...
      cos(atan2(X(end/2+1:end,:),X(1:end/2,:)))./sqrt(X(1:end/2,:).^2+X(end/2+1:end,:).^2)]*speed;
  end
end % setBgFlow