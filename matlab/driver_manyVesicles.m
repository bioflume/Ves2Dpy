clear; clc;

% LOAD THE NETWORK PATHS and PYTHON ENVIRONMENT
addpath ./shannets/

pathofDocument = fileparts(which('tension_advect_allModes_predict2024Oct.py'));
if count(py.sys.path,pathofDocument) == 0
    insert(py.sys.path,int32(0),pathofDocument);
end

pathofDocument = fileparts(which('Net_ves_merge_nocoords_nearFourier_disth.py'));
if count(py.sys.path,pathofDocument) == 0
    insert(py.sys.path,int32(0),pathofDocument);
end


pe = pyenv('Version', '/Users/gokberk/opt/anaconda3/envs/mattorch/bin/python');

% load curve_py
oc = curve_py;

% file name
fileName = ['./output/test.bin']; % to save simulation data

% Flow specification
bgFlow = 'shear';
speed = 2000; 
vinf = setBgFlow(bgFlow, speed);

% time stepping
dt = 1e-5; % time step size
Th = 0.01; % time horizon

% vesicle discretization
N = 128; % number of points to discretize vesicle (default 128 as networks)

% vesicle initialization
nv = 2;  % number of vesicles
load shearIC.mat
X = Xic;
for it = 1 : 5
  X = oc.redistributeArcLength(X);
end
X = oc.alignCenterAngle(Xic,X);

% build mlarm class to take time steps using networks
% load the normalization (mean, std) values for the networks

load ./shannets/mergedAdv_NormParams.mat % advection (merged net)
advNetInputNorm = in_param;
advNetOutputNorm = out_param;

load ./shannets/nearInterp_128modes_disth_params.mat % near-field network
nearNetInputNorm = in_param;
nearNetOutputNorm = out_param;

load ./shannets/tensionAdv_NormParams_2024Oct.mat % advection tension net
tenAdvNetInputNorm = in_param;
tenAdvNetOutputNorm = out_param;

% tension-self network
tenSelfNetInputNorm = [0.00017108717293012887; 0.06278623640537262; ...
    0.002038202714174986; 0.13337858021259308];
tenSelfNetOutputNorm = [337.7627868652344; 466.6429138183594];

% relaxation network
relaxNetInputNorm = [-8.430413700466488e-09; 0.06278684735298157; ...
    6.290720477863943e-08; 0.13339413702487946];
relaxNetOutputNorm = [-2.884585348361668e-10; 0.00020574081281665713; ...
    -5.137390512999218e-10; 0.0001763451291481033];

mlarm = MLARM_ManyFree_py(dt,vinf,oc,advNetInputNorm,...
        advNetOutputNorm,relaxNetInputNorm,relaxNetOutputNorm,...
        nearNetInputNorm,nearNetOutputNorm,tenSelfNetInputNorm,...
        tenSelfNetOutputNorm,tenAdvNetInputNorm,tenAdvNetOutputNorm);

[~,mlarm.area0,mlarm.len0] = oc.geomProp(X);

% save the initial data 
fid = fopen(fileName,'w');
fwrite(fid, [N;nv;X(:)], 'double');
fclose(fid);

% evolve in time
time = 0; it = 0; tension = zeros(N,nv);
while time < Th

  % take a time step
  tStart = tic;
  [X, tension] = mlarm.time_step(X,tension); 
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