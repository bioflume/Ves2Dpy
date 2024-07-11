classdef MLARM_ManyFree_py

properties
dt
vinf
oc
kappa
advNetInputNorm
advNetOutputNorm
relaxNetInputNorm
relaxNetOutputNorm
nearNetInputNorm
nearNetOutputNorm
tenSelfNetInputNorm
tenSelfNetOutputNorm
tenAdvNetInputNorm
tenAdvNetOutputNorm
end

methods
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function o = MLARM_ManyFree_py(dt,vinf,oc,advNetInputNorm,...
        advNetOutputNorm,relaxNetInputNorm,relaxNetOutputNorm,...
        nearNetInputNorm,nearNetOutputNorm,tenSelfNetInputNorm,...
        tenSelfNetOutputNorm,tenAdvNetInputNorm,tenAdvNetOutputNorm)
o.dt = dt; % time step size
o.vinf = vinf; % background flow (analytic -- input as function of vesicle config)
o.oc = oc; % curve class
o.kappa = 1; % bending stiffness is 1 for our simulations
% Normalization values for advection (translation) networks
o.advNetInputNorm = advNetInputNorm;
o.advNetOutputNorm = advNetOutputNorm;

% Normalization values for relaxation network
o.relaxNetInputNorm = relaxNetInputNorm;
o.relaxNetOutputNorm = relaxNetOutputNorm;

% Normalization values for near field networks
o.nearNetInputNorm = nearNetInputNorm;
o.nearNetOutputNorm = nearNetOutputNorm;

% Normalization values for tension-self network
o.tenSelfNetInputNorm = tenSelfNetInputNorm;
o.tenSelfNetOutputNorm = tenSelfNetOutputNorm;

% Normalization values for tension-advection networks
o.tenAdvNetInputNorm = tenAdvNetInputNorm;
o.tenAdvNetOutputNorm = tenAdvNetOutputNorm;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Xnew,tenNew] = time_step(o,Xold,tenOld)
oc = o.oc;

% background velocity on vesicles
vback = o.vinf(Xold);

% build vesicle class at the current step
vesicle = capsules(Xold,[],[],o.kappa,1,0);
nv = vesicle.nv;
N = vesicle.N;

% Compute bending forces + old tension forces
fBend = vesicle.tracJump(Xold,zeros(N,nv));
fTen = vesicle.tracJump(zeros(2*N,nv),tenOld);
tracJump = fBend+fTen; % total elastic force
% -----------------------------------------------------------
% 1) Explicit Tension at the Current Step

% Calculate velocity induced by vesicles on each other due to elastic force
% use neural networks to calculate near-singular integrals
farFieldtracJump = o.computeStokesInteractions(vesicle, tracJump, op, oc);

% Solve for tension
vBackSolve = o.invTenMatOnVback(Xold, vback + farFieldtracJump);
selfBendSolve = o.invTenMatOnSelfBend(Xold);
tenNew = -(vBackSolve + selfBendSolve);

% update the elastic force with the new tension
fTen = vesicle.tracJump(zeros(2*N,nv), tenNew); 
tracJump = fBend + fTen;
% -----------------------------------------------------------

% Calculate far-field again and correct near field before advection
% use neural networks to calculate near-singular integrals
farFieldtracJump = o.computeStokesInteractions(vesicle, tracJump, oc);

% Total background velocity
vbackTotal = vback + farFieldtracJump;
% -----------------------------------------------------------

% 1) COMPUTE THE ACTION OF dt*(1-M) ON Xold  
Xadv = o.translateVinfwTorch(Xold,vbackTotal);

% 2) COMPUTE THE ACTION OF RELAX OP. ON Xold + Xadv
Xnew = o.relaxWTorchNet(Xadv);    

end % DNNsolveTorchMany
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [xlayers, ylayers, velx, vely] = predictNearLayersWTorchNet(o, X, tracJump)
N = numel(X(:,1))/2;
nv = numel(X(1,:));

oc = o.oc;

in_param = o.nearNetInputNorm;
out_param = o.nearNetOutputNorm;

maxLayerDist = sqrt(1/N); % length = 1, h = 1/Nnet;

% Predictions on three layers
nlayers = 3;
dlayer = (0:nlayers-1)'/(nlayers-1) * maxLayerDist;

% standardize input
Xstand = zeros(size(X));
scaling = zeros(nv,1);
rotate = zeros(nv,1);
rotCent = zeros(2,nv);
trans = zeros(2,nv);
sortIdx = zeros(N,nv);

% Create the layers around a vesicle on which velocity calculated
tracersX = zeros(2*N,3,nv);
for k = 1 : nv
  [Xstand(:,k),scaling(k),rotate(k),rotCent(:,k),trans(:,k),sortIdx(:,k)] = o.standardizationStep(X(:,k),128);
  [~,tang] = oc.diffProp(Xstand(:,k));
  nx = tang(N+1:2*N);
  ny = -tang(1:N);

  tracersX(:,1,k) = Xstand(:,k);
  for il = 2 : nlayers 
    tracersX(:,il,k) = [Xstand(1:end/2,k)+nx*dlayer(il); Xstand(end/2+1:end,k)+ny*dlayer(il)];
  end
end

% Normalize input
input_net = zeros(nv,2,N);

for k = 1 : nv
  input_net(k,1,:) = (Xstand(1:end/2,k)-in_param(1,1))/in_param(1,2);
  input_net(k,2,:) = (Xstand(end/2+1:end,k)-in_param(1,3))/in_param(1,4);
end

% How many modes to be used
modes = [(0:N/2-1) (-N/2:-1)];
modesInUse = 16;
modeList = find(abs(modes)<=modesInUse);

% standardize tracJump
fstandRe = zeros(N, nv);
fstandIm = zeros(N, nv);
for k = 1 : nv
  fstand = o.standardize(tracJump(:,k),[0;0], rotate(k), [0;0], 1, sortIdx(:,k));
  z = fstand(1:end/2) + 1i*fstand(end/2+1:end);
  zh = fft(z);
  fstandRe(:,k) = real(zh); 
  fstandIm(:,k) = imag(zh);
end

input_conv = py.numpy.array(input_net);
[Xpredict] = pyrunfile("near_vel_predict.py","output_list",input_shape=input_conv,num_ves=py.int(nv));

% initialize ouputs
for k = 1 : nv
  velx_real{k} = zeros(N,N,3);
  vely_real{k} = zeros(N,N,3);
  velx_imag{k} = zeros(N,N,3);
  vely_imag{k} = zeros(N,N,3);
end

% denormalize output
for ij = 1 : numel(modList)
  imode = modeList(ij);
  pred = double(Xpredict{ij});
  % its size is (nv x 12 x 128) 
  % channel 1-3: vx_real_layers 0, 1, 2
  % channel 4-6; vy_real_layers 0, 1, 2
  % channel 7-9: vx_imag_layers 0, 1, 2
  % channel 10-12: vy_imag_layers 0, 1, 2

  % denormalize output
  for k = 1 : nv
    for ic = 1 : 3
      velx_real{k}(:,imode,ic) = (pred(k,0+ic,:)*out_param(imode,2,0+ic))  + out_param(imode,1,0+ic);
      vely_real{k}(:,imode,ic) = (pred(k,3+ic,:)*out_param(imode,2,3+ic))  + out_param(imode,1,3+ic);
      
      velx_imag{k}(:,imode,ic) = (pred(k,6+ic,:)*out_param(imode,2,6+ic))  + out_param(imode,1,6+ic);
      vely_imag{k}(:,imode,ic) = (pred(k,9+ic,:)*out_param(imode,2,9+ic))  + out_param(imode,1,9+ic);
    end
  end
end


velx = zeros(N,3,nv); 
vely = zeros(N,3,nv);
xlayers = zeros(N,3,nv);
ylayers = zeros(N,3,nv);
for k = 1 : nv
  velx_stand = zeros(N,3);
  vely_stand = zeros(N,3);
  for il = 1 : 3
     velx_stand(:,il) = velx_real{k}(:,:,il) * fstandRe(:,k) + velx_imag{k}(:,:,il)*fstandIm(:,k);
     vely_stand(:,il) = vely_real{k}(:,:,il) * fstandRe(:,k) + vely_imag{k}(:,:,il)*fstandIm(:,k);
     
     vx = zeros(N,1);
     vy = zeros(N,1);
     
     % destandardize
     vx(sortIdx(:,k)) = velx_stand(:,il);
     vy(sortIdx(:,k)) = vely_stand(:,il);

     VelBefRot = [vx;vy];
     
     VelRot = o.rotationOperator(VelBefRot, -rotate(k), [0;0]);
     velx(:,il,k) = VelRot(1:end/2); vely(:,il,k) = VelRot(end/2+1:end);
        
     Xl = o.destandardize(tracersX(:,il,k),trans(:,k),rotate(k),rotCent(:,k),scaling(k),sortIdx(:,k));
     xlayers(:,il,k) = Xl(1:end/2);
     ylayers(:,il,k) = Xl(end/2+1:end);
  end
end


end % predictNearLayersWTorchNet
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function farField = computeStokesInteractions(o,vesicle, tracJump, oc)

disp('Near-singular interaction through interpolation and network')
N = vesicle.N;
nv = vesicle.nv;
maxLayerDist = sqrt(vesicle.length/vesicle.N);

% Tangent
[~,tang] = oc.diffProp(vesicle.X);
% Normal
nx = tang(N+1:2*N,:);
ny = -tang(1:N,:);

xvesicle = vesicle.X(1:end/2,:); yvesicle = vesicle.X(end/2+1:end,:);

% Compute near/far hydro interactions without any correction
% First calculate the far-field
farField = zeros(2*N,nv);
for k = 1 : nv
  K = [(1:k-1) (k+1:nv)];
  farField(:,k) = o.exactStokesSL(vesicle, tracJump, vesicle.X(:,k), K);
end

% find the outermost layers of all vesicles, then perform Laplace kernel
Xlarge = zeros(2*vesicle.N,nv);
for k = 1 : nv
Xlarge(:,k) = [xvesicle(:,k)+nx(:,k)*maxLayerDist; yvesicle(:,k)+ny(:,k)*maxLayerDist];  
end

% Ray Casting to find near field
iCallNear = zeros(nv,1);
for j = 1 : nv % loop over the outermost layer around each vesicle
  % vesicles other than the j
  K = [(1:j-1) (j+1:nv)];
  % Reorder points
  S = zeros(2*vesicle.N,1);
  S(1:2:end) = Xlarge(1:end/2,j);
  S(2:2:end) = Xlarge(end/2+1:end,j);

  for k = K
    queryX{k} = []; % k's points in j's inside
    idsInStore{k} = [];

    % also store which vesicle contains k's points
    nearVesIds{k} = [];

    cnt = 1; % counter
    for p = 1 : vesicle.N
      flag = rayCasting([xvesicle(p,k);yvesicle(p,k)],S);  
      if flag
        idsInStore{k}(cnt,1) = p;
        % points where we need interpolation  
        queryX{k}(1,cnt) = xvesicle(p,k);
        queryX{k}(2,cnt) = yvesicle(p,k);
        nearVesIds{k}(cnt,1) = j; 
        cnt = cnt + 1;
        iCallNear(k) = 1;    
      end
    end
  end
end

% if needed to call near-singular correction:
if any(iCallNear)
  [xlayers, ylayers, velx, vely] = o.predictNearLayersWTorchNet(vesicle.X, tracJump);

  for k = 1 : nv 
    if iCallNear(k)
    idsIn = idsInStore{k};
    pointsIn = queryX{k};
    vesId = unique(nearVesIds{k});
    

    % layers around the vesicle j 
    Xin = [reshape(xlayers(:,:,vesId),1,3*N); reshape(ylayers(:,:,vesId),1,3*N)];
    velXInput = reshape(velx(:,:,vesId), 1, 3*N); 
    velYInput = reshape(vely(:,:,vesId), 1, 3*N);  
  
    opX = rbfcreate(Xin,velXInput,'RBFFunction','linear');
    opY = rbfcreate(Xin,velYInput,'RBFFunction','linear');

    % interpolate for the kth vesicle's points near to the Kth vesicle
    rbfVelX = rbfinterp(pointsIn, opX);
    rbfVelY = rbfinterp(pointsIn, opY);
  
    % replace the interpolated one with the direct calculation
    farX = farField(1:end/2,k); farY = farField(end/2+1:end,k);
    farX(idsIn) = rbfVelX;
    farY(idsIn) = rbfVelY;
    farField(:,k) = [farX; farY];
    end
  end % end if any(idsIn)

end % for k = 1 : nv


end % computeStokesInteractions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Xnew = translateVinfwTorch(o,Xold,vinf)
% Xinput is equally distributed in arc-length
% Xold as well. So, we add up coordinates of the same points.
N = numel(Xold(:,1))/2;
nv = numel(Xold(1,:));
oc = o.oc;

in_param = o.advNetInputNorm;
out_param = o.advNetOutputNorm;

% If we only use some modes
modes = [(0:N/2-1) (-N/2:-1)];
modesInUse = 16;
modeList = find(abs(modes)<=modesInUse);

% Standardize input
Xstand = zeros(size(Xold));
scaling = zeros(nv,1);
rotate = zeros(nv,1);
rotCent = zeros(2,nv);
trans = zeros(2,nv);
sortIdx = zeros(N,nv);
for k = 1 : nv
  [Xstand(:,k),scaling(k),rotate(k),rotCent(:,k),trans(:,k),sortIdx(:,k)] = o.standardizationStep(Xold(:,k),N);
end


% Normalize input
input_list = []; 
cnt = 1;
for imode = modeList
  if imode ~= 1
  input_net = zeros(nv,2,128);  
  x_mean = in_param(imode-1,1);
  x_std = in_param(imode-1,2);
  y_mean = in_param(imode-1,3);
  y_std = in_param(imode-1,4);
  for k = 1 : nv
    input_net(k,1,:) = (Xstand(1:end/2,k)-x_mean)/x_std;
    input_net(k,2,:) = (Xstand(end/2+1:end,k)-y_mean)/y_std;
  end
  input_list{cnt} = py.numpy.array(input_net);
  cnt = cnt + 1;
  end
end % imode


tS = tic;
[Xpredict] = pyrunfile("advect_predict.py","output_list",input_shape=input_list,num_ves=py.int(nv));
tPyCall = toc(tS);

disp(['Calling python to predict MV takes ' num2str(tPyCall) ' seconds'])
% we have 128 modes
% Approximate the multiplication M*(FFTBasis)     
Z11r = zeros(N,N,nv); Z12r = Z11r;
Z21r = Z11r; Z22r = Z11r;

tS = tic;
for ij = 1 : numel(modeList)-1
  
  imode = modeList(ij+1); % mode index # skipping the first mode
  pred = double(Xpredict{ij}); % size(pred) = [nv 2 256]


  % denormalize output
  real_mean = out_param(imode-1,1);
  real_std = out_param(imode-1,2);
  imag_mean = out_param(imode-1,3);
  imag_std = out_param(imode-1,4);
  
  for k = 1 : nv
    % first channel is real
    pred(k,1,:) = (pred(k,1,:)*real_std) + real_mean;
    % second channel is imaginary
    pred(k,2,:) = (pred(k,2,:)*imag_std) + imag_mean;

    Z11r(:,imode,k) = pred(k,1,1:end/2);
    Z21r(:,imode,k) = pred(k,1,end/2+1:end);
    Z12r(:,imode,k) = pred(k,2,1:end/2);
    Z22r(:,imode,k) = pred(k,2,end/2+1:end);
  end
end
tOrganize = toc(tS);
disp(['Organizing MV output takes ' num2str(tOrganize) ' seconds'])

% Take fft of the velocity (should be standardized velocity)
% only sort points and rotate to pi/2 (no translation, no scaling)
Xnew = zeros(size(Xold));
for k = 1 : nv
  vinfStand = o.standardize(vinf(:,k),[0;0],rotate(k),[0;0],1,sortIdx(:,k));
  z = vinfStand(1:end/2)+1i*vinfStand(end/2+1:end);

  zh = fft(z);
  V1 = real(zh); V2 = imag(zh);
  % Compute the approximate value of the term M*vinf
  MVinfStand = [Z11r(:,:,k)*V1+Z12r(:,:,k)*V2; Z21r(:,:,k)*V1+Z22r(:,:,k)*V2];
  % Need to destandardize MVinf (take sorting and rotation back)
  MVinf = zeros(size(MVinfStand));
  MVinf([sortIdx(:,k);sortIdx(:,k)+N]) = MVinfStand;
  MVinf = o.rotationOperator(MVinf,-rotate(k),[0;0]);

  Xnew(:,k) = Xold(:,k) + o.dt * vinf(:,k) - o.dt*MVinf;
end

% XnewStand = Xstand + o.dt*vinfStand - o.dt*MVinf;
% Update the position
% Xnew = o.destandardize(XnewStand,trans,rotate,scaling,sortIdx);
end % translateVinfwTorch
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Xnew = relaxWTorchNet(o,Xmid)  

% 1) RELAXATION w/ NETWORK
% Standardize vesicle Xmid
Xin = zeros(size(Xmid));
nv = numel(Xmid(1,:));
N = numel(Xmid(:,1))/2;

scaling = zeros(nv,1); rotate = zeros(nv,1); 
rotCent = zeros(2,nv); trans = zeros(2,nv);
sortIdx = zeros(N,nv);

for k = 1 : nv
  [Xin(:,k),scaling(k),rotate(k),rotCent(:,k),trans(:,k),sortIdx(:,k)] = ...
    o.standardizationStep(Xmid(:,k),128);
end

% INPUT NORMALIZATION INFO

% Normalize input
x_mean = o.relaxNetInputNorm(1);
x_std = o.relaxNetInputNorm(2);
y_mean = o.relaxNetInputNorm(3);
y_std = o.relaxNetInputNorm(4);

% INPUT NORMALIZING

% REAL SPACE
Xstand = Xin; % before normalization
Xin(1:end/2,:) = (Xin(1:end/2,:)-x_mean)/x_std;
Xin(end/2+1:end,:) = (Xin(end/2+1:end,:)-y_mean)/y_std;
XinitShape = zeros(nv,2,128);
for k = 1 : nv
XinitShape(k,1,:) = Xin(1:end/2,k)'; 
XinitShape(k,2,:) = Xin(end/2+1:end,k)';
end
XinitConv = py.numpy.array(XinitShape);


% OUTPUT

% June8 - Dt1E5
[DXpredictStand] = pyrunfile("relax_predict_DIFF_June8_dt1E5.py", "predicted_shape", input_shape=XinitConv);


% For the 625k - June8 - Dt = 1E-5 data
x_mean = o.relaxNetOutputNorm(1);
x_std = o.relaxNetOutputNorm(2);
y_mean = o.relaxNetOutputNorm(3);
y_std = o.relaxNetOutputNorm(4);


DXpred = zeros(size(Xin));
DXpredictStand = double(DXpredictStand);
Xnew = zeros(size(Xmid));

for k = 1 : nv
% normalize output
DXpred(1:end/2,k) = DXpredictStand(k,1,:)*x_std + x_mean;
DXpred(end/2+1:end,k) = DXpredictStand(k,2,:)*y_std + y_mean;


DXpred(:,k) = DXpred(:,k)/1E-5 * o.dt; % scale the output if dt is other than 1E-5
Xpred = Xstand(:,k) + DXpred(:,k);

Xnew(:,k) = o.destandardize(Xpred,trans(:,k),rotate(k),rotCent(:,k),...
    scaling(k),sortIdx(:,k));
end



end % relaxWTorchNet
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function vBackSolve = invTenMatOnVback(o,X,vinf)
% Approximate inv(Div*G*Ten)*Div*vExt 
% input X is non-standardized
    
% number of vesicles
nv = numel(X(1,:));
% number of points of exact solve
N = numel(X(:,1))/2;    
% number of points for network

% Modes to be called
modes = [(0:N/2-1) (-N/2:-1)];
modesInUse = 16;
modeList = find(abs(modes)<=modesInUse);

% Standardize vesicle Xmid
Xstand = zeros(size(X));

scaling = zeros(nv,1); rotate = zeros(nv,1); 
rotCent = zeros(2,nv); trans = zeros(2,nv);
sortIdx = zeros(N,nv);

for k = 1 : nv
  [Xstand(:,k),scaling(k),rotate(k),rotCent(:,k),trans(:,k),sortIdx(:,k)] = ...
    o.standardizationStep(X(:,k),128);
end

in_param = o.tenAdvNetInputNorm;
out_param = o.tenAdvNetOutputNorm;

% Normalize input
input_list = []; 
cnt = 1;
for imode = modeList
  if imode ~= 1
  input_net = zeros(nv,2,128);  
  x_mean = in_param(imode-1,1);
  x_std = in_param(imode-1,2);
  y_mean = in_param(imode-1,3);
  y_std = in_param(imode-1,4);
  for k = 1 : nv
    input_net(k,1,:) = (Xstand(1:end/2,k)-x_mean)/x_std;
    input_net(k,2,:) = (Xstand(end/2+1:end,k)-y_mean)/y_std;
  end
  input_list{cnt} = py.numpy.array(input_net);
  cnt = cnt + 1;
  end
end % imode

[Xpredict] = pyrunfile("tension_advect_predict.py","output_list",input_shape=input_list,num_ves=py.int(nv));

% Approximate the multiplication Z = inv(DivGT)DivPhi_k
Z1 = zeros(N,N,nv); Z2 = Z11r;


for ij = 1 : numel(modeList)-1
  
  imode = modeList(ij+1); % mode index # skipping the first mode
  pred = double(Xpredict{ij}); % size(pred) = [1 2 256]


  % denormalize output
  real_mean = out_param(imode-1,1);
  real_std = out_param(imode-1,2);
  imag_mean = out_param(imode-1,3);
  imag_std = out_param(imode-1,4);
  
  for k = 1 : nv
    % first channel is real
    pred(k,1,:) = (pred(k,1,:)*real_std) + real_mean;
    % second channel is imaginary
    pred(k,2,:) = (pred(k,2,:)*imag_std) + imag_mean;

    Z1(:,imode,k) = pred(k,1,:);
    Z2(:,imode,k) = pred(k,2,:);
  end
end

vBackSolve = zeros(N,nv);
for k = 1 : nv
  % Take fft of the velocity, standardize velocity
  vinfStand = o.standardize(vinf(:,k),[0;0],rotate(k),1,sortIdx(:,k));
  z = vinfStand(1:end/2)+1i*vinfStand(end/2+1:end);
  zh = fft(z);
  V1 = real(zh); V2 = imag(zh);

  % Compute the approximation to inv(Div*G*Ten)*Div*vExt
  MVinfStand = (Z1(:,:,k)*V1+Z2(:,:,k)*V2);

  % Destandardize the multiplication
  MVinf = zeros(size(MVinfStand));
  MVinf([sortIdx(:,k);sortIdx(:,k)+N]) = MVinfStand;
  vBackSolve(:,k) = o.rotationOperator(MVinf,-rotate(:,k),[0;0]);
end
end % invTenMatOnVback
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function tenPred = invTenMatOnSelfBend(o,X)
% Approximate inv(Div*G*Ten)*G*(-Ben)*x

% number of vesicles
Xstand = zeros(size(X));
nv = numel(X(1,:));
N = numel(X(:,1))/2;

scaling = zeros(nv,1); rotate = zeros(nv,1); 
rotCent = zeros(2,nv); trans = zeros(2,nv);
sortIdx = zeros(N,nv);

for k = 1 : nv
  [Xstand(:,k),scaling(k),rotate(k),rotCent(:,k),trans(:,k),sortIdx(:,k)] = ...
    o.standardizationStep(X(:,k),128);
end

% Normalize input
x_mean = o.tenSelfNetInputNorm(1);
x_std = o.tenSelfNetInputNorm(2);
y_mean = o.tenSelfNetInputNorm(3);
y_std = o.tenSelfNetInputNorm(4);

% Adjust the input shape for the network
XinitShape = zeros(nv,2,128);
for k = 1 : nv
  XinitShape(k,1,:) = (Xstand(1:end/2,k)'-x_mean)/x_std; 
  XinitShape(k,2,:) = (Xstand(end/2+1:end,k)'-y_mean)/y_std;
end
XinitConv = py.numpy.array(XinitShape);

% Make prediction -- needs to be adjusted for python
[tenPredictStand] = pyrunfile("tension_self_network.py", "predicted_shape", input_shape=XinitConv);

% Denormalize output
out_mean = o.tenSelfNetOutputNorm(1);
out_std = o.tenSelfNetOutputNorm(2);


tenPred = zeros(N,nv);
tenPredictStand = double(tenPredictStand);

for k = 1 : nv 
  % also destandardize
  tenPred(sortIdx(:,k)) = (tenPredictStand(k,1,:)*out_std + out_mean)...
      /scaling(k)^2;
end

end % invTenMatOnSelfBend
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [X,scaling,rotate,rotCent,trans,sortIdx] = standardizationStep(o,Xin,Nnet)
oc = o.oc;
N = numel(Xin)/2;
if Nnet ~= N
  Xin = [interpft(Xin(1:end/2),Nnet);interpft(Xin(end/2+1:end),Nnet)];    
end

% Equally distribute points in arc-length
for iter = 1 : 10
  [Xin,~,~] = oc.redistributeArcLength(Xin);
end


X = Xin;
[trans,rotate,rotCent,scaling,sortIdx] = o.referenceValues(X);

% Fix misalignment in center and angle due to reparametrization
% X = oc.alignCenterAngle(Xin,X);

% standardize angle, center, scaling and point order

X = o.standardize(X,trans,rotate,rotCent,scaling,sortIdx);
end % standardizationStep
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function XrotSort = standardize(o,X,translation,rotation,rotCent,scaling,sortIdx)
N = numel(sortIdx);

% translate, rotate and scale configuration

Xrotated = o.rotationOperator(X,rotation,rotCent);   
Xrotated = o.translateOp(Xrotated,translation);

% now order the points
XrotSort = [Xrotated(sortIdx);Xrotated(sortIdx+N)];

XrotSort = scaling*XrotSort;

end % standardize
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function X = destandardize(o,XrotSort,translation,rotation,rotCent,scaling,sortIdx)

N = numel(sortIdx);    
    
% scaling back
XrotSort = XrotSort/scaling;

% change ordering back 
X = zeros(size(XrotSort));
X([sortIdx;sortIdx+N]) = XrotSort;

% take translation back
X = o.translateOp(X,-translation);

% take rotation back
X = o.rotationOperator(X,-rotation,rotCent);


end % destandardize

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [translation,rotation,rotCent,scaling,sortIdx] = referenceValues(o,Xref)
oc = o.oc;
N = numel(Xref)/2;

% find translation, rotation and scaling
center = oc.getPhysicalCenterShan(Xref);
V = oc.getPrincAxesGivenCentroid(Xref,center);
% % find rotation angle
w = [0;1]; % y-axis
rotation = atan2(w(2)*V(1)-w(1)*V(2), w(1)*V(1)+w(2)*V(2));

% find the ordering of the points
rotCent = center;
Xref = o.rotationOperator(Xref, rotation, center);
center = oc.getPhysicalCenterShan(Xref);
translation = -center;

Xref = o.translateOp(Xref, translation);

firstQuad = find(Xref(1:end/2)>=0 & Xref(end/2+1:end)>=0);
theta = atan2(Xref(end/2+1:end),Xref(1:end/2));
[~,idx]= min(theta(firstQuad));
sortIdx = [(firstQuad(idx):N)';(1:firstQuad(idx)-1)'];

% amount of scaling
[~,~,length] = oc.geomProp(Xref);
scaling = 1/length;
end % referenceValues
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Xrot = rotationOperator(o,X,theta, rotCent)
% Get x-y coordinates
Xrot = zeros(size(X));
x = X(1:end/2)-rotCent(1); y = X(end/2+1:end)-rotCent(2);

% Rotated shape
xrot = (x)*cos(theta) - (y)*sin(theta);
yrot = (x)*sin(theta) + (y)*cos(theta);

Xrot(1:end/2) = xrot+rotCent(1);
Xrot(end/2+1:end) = yrot+rotCent(2);
end % rotationOperator
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Xnew = translateOp(o,X,transXY)
Xnew = zeros(size(X));
Xnew(1:end/2) = X(1:end/2)+transXY(1);
Xnew(end/2+1:end) = X(end/2+1:end)+transXY(2);
end  % translateOp  

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end % methods

end % dnnTools
