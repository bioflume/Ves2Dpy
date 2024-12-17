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
torchLaplaceInNorm
torchLaplaceOutNorm
area0
len0
irepulsion
repStrength
end

methods
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function o = MLARM_ManyFree_py(dt,vinf,oc,advNetInputNorm,...
        advNetOutputNorm,relaxNetInputNorm,relaxNetOutputNorm,...
        nearNetInputNorm,nearNetOutputNorm,tenSelfNetInputNorm,...
        tenSelfNetOutputNorm,tenAdvNetInputNorm,tenAdvNetOutputNorm,irepulsion)
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

% Flag for repulsion
o.irepulsion = irepulsion;
o.repStrength = 1E+3;

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Xnew,tenNew] = time_step(o,Xold,tenOld)
oc = o.oc;

% background velocity on vesicles
vback = o.vinf(Xold);

% build vesicle class at the current step
vesicle = capsules_py(Xold,[],[],o.kappa,1);
nv = vesicle.nv;
N = vesicle.N;

% Compute velocity induced by repulsion force
repForce = zeros(2*N,nv);
if o.irepulsion
  repForce = vesicle.repulsionForce(Xold,o.repStrength);
end

% Compute bending forces + old tension forces
fBend = vesicle.tracJump(Xold,zeros(N,nv));
fTen = vesicle.tracJump(zeros(2*N,nv),tenOld);
tracJump = fBend+fTen; % total elastic force
% -----------------------------------------------------------
% 1) Explicit Tension at the Current Step

% Calculate velocity induced by vesicles on each other due to elastic force
% use neural networks to calculate near-singular integrals
[velx_real, vely_real, velx_imag, vely_imag, xlayers, ylayers, transNear, rotateNear, ...
    rotCentNear, scalingNear, sortIdxNear] = o.predictNearLayers(vesicle.X); 
farFieldtracJump = o.computeStokesInteractions(vesicle, tracJump, repForce, ...
    velx_real, vely_real, velx_imag, vely_imag, xlayers, ylayers, transNear, rotateNear, ...
    rotCentNear, scalingNear, sortIdxNear);

% filter the far-field
farFieldtracJump = oc.upsThenFilterShape(farFieldtracJump,4*N,16);

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
farFieldtracJump = o.computeStokesInteractions(vesicle, tracJump, repForce,...
    velx_real, vely_real, velx_imag, vely_imag, xlayers, ylayers, transNear, rotateNear, ...
    rotCentNear, scalingNear, sortIdxNear);
% filter the far-field
farFieldtracJump = oc.upsThenFilterShape(farFieldtracJump,4*N,16);

% Total background velocity
vbackTotal = vback + farFieldtracJump;
% -----------------------------------------------------------

% 1) COMPUTE THE ACTION OF dt*(1-M) ON Xold  
Xadv = o.translateVinfwTorch(Xold,vbackTotal);

% filter shape
Xadv = oc.upsThenFilterShape(Xadv,4*N,16);

% 2) COMPUTE THE ACTION OF RELAX OP. ON Xold + Xadv
Xnew = o.relaxWTorchNet(Xadv);    

% Redistribute points equally in arc-length
XnewO = Xnew;
for it = 1 : 5
  Xnew = oc.redistributeArcLength(Xnew);
end
Xnew = oc.alignCenterAngle(XnewO,Xnew);


% area-length correction
Xnew = oc.correctAreaAndLength(Xnew,o.area0,o.len0);


% filter shape
Xnew = oc.upsThenFilterShape(Xnew,4*N,64);

end % DNNsolveTorchMany
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [velx_real, vely_real, velx_imag, vely_imag, xlayers, ylayers, trans, rotate, rotCent, scaling, sortIdx] = predictNearLayers(o, X)
% prediction of the velocity in the near-field layers (using merged net)
N = numel(X(:,1))/2;
nv = numel(X(1,:));

oc = o.oc;

in_param = o.nearNetInputNorm;
out_param = o.nearNetOutputNorm;

maxLayerDist = 1/N; % length = 1, h = 1/N;

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
  [Xstand(:,k),scaling(k),rotate(k),rotCent(:,k),trans(:,k),sortIdx(:,k)] = o.standardizationStep(X(:,k),N);
  [~,tang] = oc.diffProp(Xstand(:,k));
  nx = tang(N+1:2*N);
  ny = -tang(1:N);

  tracersX(:,1,k) = Xstand(:,k);
  for il = 2 : nlayers 
    tracersX(:,il,k) = [Xstand(1:end/2,k)+nx*dlayer(il); Xstand(end/2+1:end,k)+ny*dlayer(il)];
  end
end

% Normalize input
input_net = zeros(nv,2*N,N);

for ij = 1 : N
  for k = 1 : nv
    input_net(k,(ij-1)*2+1,:) = (Xstand(1:end/2,k)-in_param(1,1))/in_param(1,2);
    input_net(k,2*ij,:) = (Xstand(end/2+1:end,k)-in_param(1,3))/in_param(1,4);
  end
end


% How many modes to be used
modes = [(0:N/2-1) (-N/2:-1)];
modesInUse = N;
modeList = find(abs(modes)<=modesInUse);


input_conv = py.numpy.array(input_net);
[Xpredict] = pyrunfile("near_vel_allModesAth_predict.py","output_list",input_shape=input_conv);
Xpredict = double(Xpredict);

for k = 1 : nv
  velx_real{k} = zeros(N,N,3);
  vely_real{k} = zeros(N,N,3);
  velx_imag{k} = zeros(N,N,3);
  vely_imag{k} = zeros(N,N,3);
end

% denormalize output
for ij = 1 : numel(modeList)
  imode = modeList(ij);
  pred = Xpredict(:,(ij-1)*12+1:ij*12,:);
  % its size is (nv x 12 x 128) 
  % channel 1-3: vx_real_layers 0, 1, 2
  % channel 4-6; vy_real_layers 0, 1, 2
  % channel 7-9: vx_imag_layers 0, 1, 2
  % channel 10-12: vy_imag_layers 0, 1, 2

  % denormalize output
  for k = 1 : nv
    velx_real{k}(:,imode,1) = (pred(k,1,:)*out_param(imode,2,1))  + out_param(imode,1,1);
    velx_real{k}(:,imode,2) = (pred(k,2,:)*out_param(imode,2,2))  + out_param(imode,1,2);
    velx_real{k}(:,imode,3) = (pred(k,3,:)*out_param(imode,2,3))  + out_param(imode,1,3);
    vely_real{k}(:,imode,1) = (pred(k,4,:)*out_param(imode,2,4))  + out_param(imode,1,4);
    vely_real{k}(:,imode,2) = (pred(k,5,:)*out_param(imode,2,5))  + out_param(imode,1,5);
    vely_real{k}(:,imode,3) = (pred(k,6,:)*out_param(imode,2,6))  + out_param(imode,1,6);

    velx_imag{k}(:,imode,1) = (pred(k,7,:)*out_param(imode,2,7))  + out_param(imode,1,7);
    velx_imag{k}(:,imode,2) = (pred(k,8,:)*out_param(imode,2,8))  + out_param(imode,1,8);
    velx_imag{k}(:,imode,3) = (pred(k,9,:)*out_param(imode,2,9))  + out_param(imode,1,9);
    vely_imag{k}(:,imode,1) = (pred(k,10,:)*out_param(imode,2,10))  + out_param(imode,1,10);
    vely_imag{k}(:,imode,2) = (pred(k,11,:)*out_param(imode,2,11))  + out_param(imode,1,11);
    vely_imag{k}(:,imode,3) = (pred(k,12,:)*out_param(imode,2,12))  + out_param(imode,1,12);
  end
end

% outputs
% velx_real, vely_real, velx_imag, vely_imag, xlayers, ylayers, trans, rotate, rotCent, scaling, sortIdx

xlayers = zeros(N,3,nv);
ylayers = zeros(N,3,nv);
for k = 1 : nv
  for il = 1 : 3
     Xl = o.destandardize(tracersX(:,il,k),trans(:,k),rotate(k),rotCent(:,k),scaling(k),sortIdx(:,k));
     xlayers(:,il,k) = Xl(1:end/2);
     ylayers(:,il,k) = Xl(end/2+1:end);
  end
end



end % predictNearLayers
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [velx, vely] = buildVelocityInNear(o,tracJump, velx_real,...
        vely_real, velx_imag, vely_imag, trans, rotate, rotCent, scaling, sortIdx)

nv = numel(rotate);
N = numel(sortIdx(:,1));

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


velx = zeros(N,3,nv); 
vely = zeros(N,3,nv);
for k = 1 : nv
  velx_stand = zeros(N,3);
  vely_stand = zeros(N,3);
  for il = 1 : 3
     velx_stand(:,il) = velx_real{k}(:,:,il) * fstandRe(:,k) + velx_imag{k}(:,:,il)*fstandIm(:,k);
     vely_stand(:,il) = vely_real{k}(:,:,il) * fstandRe(:,k) + vely_imag{k}(:,:,il)*fstandIm(:,k);
     
     vx = zeros(N,1);
     vy = zeros(N,1);
     
     vx(sortIdx(:,k)) = velx_stand(:,il);
     vy(sortIdx(:,k)) = vely_stand(:,il);

     VelBefRot = [vx;vy];
     
     VelRot = o.rotationOperator(VelBefRot, -rotate(k), [0;0]);
     velx(:,il,k) = VelRot(1:end/2); vely(:,il,k) = VelRot(end/2+1:end);
  end
end


end % buildVelocityInNear
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function farField = computeStokesInteractions(o,vesicle, repForce, tracJump, ...
        velx_real, vely_real, velx_imag, vely_imag, xlayers, ylayers, trans,...
        rotate,rotCent, scaling, sortIdx)

N = vesicle.N;
nv = vesicle.nv;

xvesicle = vesicle.X(1:end/2,:); yvesicle = vesicle.X(end/2+1:end,:);

% Total force density on vesicles
totalForce = tracJump + repForce;

% Compute near/far hydro interactions
% with upsampling by 2
NearV2V = vesicle.getZone([],1);    
zone = NearV2V.zone;


% First calculate the far-field
farField = zeros(2*N,nv);
for k = 1 : nv
  K = [(1:k-1) (k+1:nv)];
  farField(:,k) = o.exactStokesSL(vesicle, totalForce, vesicle.X(:,k), K);
end

% Predict velocity on layers
% Get velocity on layers once predicted
[velx, vely] = o.buildVelocityInNear(totalForce, velx_real, vely_real, ...
    velx_imag, vely_imag, trans, rotate, rotCent, scaling, sortIdx);

% Get velocity due to repulsion on vesicles themselves (G*repForce)
selfRepVel = zeros(2*N,nv);
if o.irepulsion
  [selfRepVelx, selfRepVely] = o.buildVelocityInNear(repForce, velx_real, vely_real, ...
      velx_imag, vely_imag, trans, rotate, rotCent, scaling, sortIdx);
  selfRepVel = [selfRepVelx(:,:,1); selfRepVely(:,:,1)]; % velocity on the first layer
end

opX = []; opY = [];
for k = 1 : nv
 % layers around the vesicle j 
 Xin = [reshape(xlayers(:,:,k),1,3*N); reshape(ylayers(:,:,k),1,3*N)];
 velXInput = reshape(velx(:,:,k), 1, 3*N); 
 velYInput = reshape(vely(:,:,k), 1, 3*N);  
  
 opX{k} = rbfcreate(Xin,velXInput,'RBFFunction','linear');
 opY{k} = rbfcreate(Xin,velYInput,'RBFFunction','linear');
end

nearField = zeros(size(farField));

for k1 = 1 : nv
  K = [(1:k1-1) (k1+1:nv)];
  for k2 = K
    % points on vesicle k2 close to k1  
    J = find(zone{k1}(:,k2) == 1);  
    if numel(J) ~= 0
      % need tp subtract off contribution due to vesicle k1 since its layer
      % potential will be evaulated through interpolation
      potTar = o.exactStokesSL(vesicle, totalForce, [xvesicle(J,k2);yvesicle(J,k2)],k1);
      nearField(J,k2) = nearField(J,k2) - potTar(1:numel(J));
      nearField(J+N,k2) = nearField(J+N,k2) - potTar(numel(J)+1:end);
         
      % now interpolate
      for i = 1 : numel(J)
        pointsIn = [xvesicle(J(i),k2);yvesicle(J(i),k2)];
        % interpolate for the k2th vesicle's points near to the k1th vesicle
        rbfVelX = rbfinterp(pointsIn, opX{k1});
        rbfVelY = rbfinterp(pointsIn, opY{k1});
        nearField(J(i),k2) = nearField(J(i),k2) + rbfVelX;
        nearField(J(i)+N,k2) = nearField(J(i)+N,k2) + rbfVelY; 
      end
    end % if numel(J)

  end % for k2
end % for k1



% finally add the corrected nearfield to the farfield
farField = farField + nearField + selfRepVel;

end % computeStokesInteractions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Xnew = translateVinfwTorch(o,Xold,vinf)

% Xinput is equally distributed in arc-length
% Xold as well. So, we add up coordinates of the same points.
% Uses merged network

N = numel(Xold(:,1))/2;
nv = numel(Xold(1,:));
oc = o.oc;

theta = (0:N-1)'/N*2*pi;
ks = (0:N-1)';
basis = 1/N*exp(1i*theta*ks');

modes = [(0:N/2-1) (-N/2:-1)];
modesInUse = N;
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

in_param = o.advNetInputNorm;
out_param = o.advNetOutputNorm;

% input normalization
x_mean = in_param(1,1);
x_std = in_param(1,2);
y_mean = in_param(1,3);
y_std = in_param(1,4);

% Normalize input
input_net = zeros(nv,2*(N-1),2*N);  
for ij = 1 : N-1
  for k = 1 : nv
    input_net(k,2*(ij-1)+1,1:N) = (Xstand(1:end/2,k)-x_mean)/x_std;
    input_net(k,2*(ij-1)+1,N+1:2*N) = (Xstand(end/2+1:end,k)-y_mean)/y_std;

    rr = real(basis(:,ij+1));
    ii = imag(basis(:,ij+1));

    input_net(k,2*ij,1:N) = rr;
    input_net(k,2*ij,N+1:2*N) = ii;
  end % imode
end

input_conv = py.numpy.array(input_net);
[Xpredict] = pyrunfile("advect_predict_merged.py","output_list",input_shape=input_conv,num_ves=py.int(nv));

allmodes_pred = double(Xpredict);

% Approximate the multiplication M*(FFTBasis)     
Z11r = zeros(N,N,nv); Z12r = Z11r;
Z21r = Z11r; Z22r = Z11r;


for ij = 1 : N-1
  
  imode = modeList(ij+1); % mode index # skipping the first mode
  pred = allmodes_pred(:,2*(ij-1)+1:2*ij,:); % size(pred) = [1 2 256]


  % denormalize output
  real_mean = out_param(ij,1);
  real_std = out_param(ij,2);
  imag_mean = out_param(ij,3);
  imag_std = out_param(ij,4);
  
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


% Take fft of the velocity (should be standardized velocity)
% only sort points and rotate to pi/2 (no translation, no scaling)
Xnew = zeros(size(Xold));
MVinfStore = Xnew;
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
  MVinfStore(:,k) = MVinf;  
  Xnew(:,k) = Xold(:,k) + o.dt * vinf(:,k) - o.dt*MVinf;   
end


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
    o.standardizationStep(Xmid(:,k),N);
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
function tension = invTenMatOnVback(o,X,vinf)

% Approximate inv(Div*G*Ten)*Div*vExt 
% input X is non-standardized
N = numel(X(:,1))/2;
nv = numel(X(1,:));
oc = o.oc;

% Modes to be called
modes = [(0:N/2-1) (-N/2:-1)];
modesInUse = N;
modeList = find(abs(modes)<=modesInUse);

% Standardize input
Xstand = zeros(size(X));
scaling = zeros(nv,1);
rotate = zeros(nv,1);
rotCent = zeros(2,nv);
trans = zeros(2,nv);
sortIdx = zeros(N,nv);

for k = 1 : nv
  [Xstand(:,k),scaling(k),rotate(k),rotCent(:,k),trans(:,k),sortIdx(:,k)] = o.standardizationStep(X(:,k),N);
end

in_param = o.tenAdvNetInputNorm;
out_param = o.tenAdvNetOutputNorm;

% Normalize input
input_net = zeros(nv,2*(N-1),N);

for imode = 1 : N-1 
  for k = 1 : nv
    input_net(k,2*(imode-1)+1,:) = (Xstand(1:end/2,k)-in_param(1,1))/in_param(1,2);
    input_net(k,2*imode,:) = (Xstand(end/2+1:end,k)-in_param(1,3))/in_param(1,4);
  end
end


input_conv = py.numpy.array(input_net);
[Xpredict] = pyrunfile("tension_advect_allModes_predict2024Oct.py","output_list",input_shape=input_conv,num_ves=py.int(nv),modesInUse=py.int(modesInUse));

% Approximate the multiplication M*(FFTBasis)     
Z1 = zeros(N,N,nv); Z2 = Z1;

pred = double(Xpredict); % size (nv, 2*(N-1), N)
for ij = 1 : numel(modeList)-1
  
  imode = modeList(ij+1); % mode index # skipping the first mode
  
  % denormalize output
  real_mean = out_param(ij,1);
  real_std = out_param(ij,2);
  imag_mean = out_param(ij,3);
  imag_std = out_param(ij,4);
  
  for k = 1 : nv
    % first channel is real
    Z1(:,imode,k) = (pred(k,2*(ij-1)+1,:)*real_std) + real_mean;
    % second channel is imaginary
    Z2(:,imode,k) = (pred(k,2*ij,:)*imag_std) + imag_mean;
  end
end

% Take fft of the velocity (should be standardized velocity)
% only sort points and rotate to pi/2 (no translation, no scaling)
tension = zeros(N,nv);
for k = 1 : nv
  vinfStand = o.standardize(vinf(:,k),[0;0],rotate(k),[0;0],1,sortIdx(:,k));
  z = vinfStand(1:end/2)+1i*vinfStand(end/2+1:end);

  zh = fft(z);
  V1 = real(zh); V2 = imag(zh);
  % Compute the approximate value of the term M*vinf
  MVinfStand = Z1(:,:,k)*V1 + Z2(:,:,k)*V2;
  % Need to destandardize MVinf (take sorting and rotation back)
  MVinf = zeros(size(MVinfStand));
  MVinf(sortIdx(:,k)) = MVinfStand;
  tension(:,k) = MVinf;
end

end % invTenMatOnVback
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function tension = invTenMatOnSelfBend(o,X)
% Approximate inv(Div*G*Ten)*G*(-Ben)*x

% Xinput is equally distributed in arc-length
% Xold as well. So, we add up coordinates of the same points.
N = numel(X(:,1))/2;
nv = numel(X(1,:));
oc = o.oc;


scaling = zeros(nv,1); rotate = zeros(nv,1); 
rotCent = zeros(2,nv); trans = zeros(2,nv);
sortIdx = zeros(N,nv);
Xin = zeros(size(X));

for k = 1 : nv
  [Xin(:,k),scaling(k),rotate(k),rotCent(:,k),trans(:,k),sortIdx(:,k)] = ...
    o.standardizationStep(X(:,k),N);
end

x_mean = o.tenSelfNetInputNorm(1);
x_std = o.tenSelfNetInputNorm(2);
y_mean = o.tenSelfNetInputNorm(3);
y_std = o.tenSelfNetInputNorm(4);

out_mean = o.tenSelfNetOutputNorm(1);
out_std = o.tenSelfNetOutputNorm(2);

Xin(1:end/2,:) = (Xin(1:end/2,:)-x_mean)/x_std;
Xin(end/2+1:end,:) = (Xin(end/2+1:end,:)-y_mean)/y_std;
XinitShape = zeros(nv,2,N);
for k = 1 : nv
XinitShape(k,1,:) = Xin(1:end/2,k)'; 
XinitShape(k,2,:) = Xin(end/2+1:end,k)';
end
XinitConv = py.numpy.array(XinitShape);

[Xpredict] = pyrunfile("self_tension_solve_2024Oct.py","predicted_shape",input_shape=XinitConv);

tenStand = double(Xpredict);
tension = zeros(N,nv);

for k = 1 : nv
  tenOut = tenStand(k,1,:)*out_std + out_mean;
  tension(sortIdx(:,k),k) = tenOut/scaling(k)^2;
end



end % invTenMatOnSelfBend
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function icollisionVes = collisionDetection(o, X)
N = numel(X(:,1))/2;
nv = numel(X(1,:));

oc = o.oc;
in_param = o.torchLaplaceInNorm;
out_param = o.torchLaplaceOutNorm;

nlayers = 5; % 4 of them are predicted -- 
% Laplace integral on the layer aligning with vesicle is known and zero
dlayer = [-1; -1/2; 0; 1; 2] * 1/N;
% can cheat here because we know that the double-layer
% potential applied to our function f will always be 0
% This won't work if we are considering density functions
% that are not one everywhere

% Density function is constant.  Pad second half of it with zero
f = [ones(N,nv);zeros(N,nv)];
% standardize input
Xstand = zeros(size(X));
nv = numel(X(1,:));
scaling = zeros(nv,1);
rotate = zeros(nv,1);
rotCent = zeros(2,nv);
trans = zeros(2,nv);
sortIdx = zeros(Nnet,nv);

tracersX = zeros(2*N,5,nv);

for k = 1 : nv
  [Xstand(:,k),scaling(k),rotate(k),rotCent(:,k),trans(:,k),sortIdx(:,k)] = o.standardizationStep(X(:,k),N);
  [~,tang] = oc.diffProp(Xstand(:,k));
  nx = tang(N+1:2*N);
  ny = -tang(1:N);

  for il = 1 : nlayers 
    tracersX(:,il,k) = [Xstand(1:end/2,k)+nx*dlayer(il); Xstand(end/2+1:end,k)+ny*dlayer(il)];
  end
end

% Normalize input
input_net = zeros(nv,2*N);

for k = 1 : nv
  input_net(k,1,:) = (Xstand(1:end/2,k)-in_param(1,1))/in_param(1,2);
  input_net(k,2,:) = (Xstand(end/2+1:end,k)-in_param(1,3))/in_param(1,4);
end


input_conv = py.numpy.array(input_net);
[Xpredict] = pyrunfile("laplaceIntegral_predict.py","output_list",input_shape=input_conv);

Xpredict = double(Xpredict); % let's say the output is (nv x 4 x N)
laplaceDL = zeros(N,nv,5);
% denormalize output
for k = 1 : nv
  laplaceDL(:,k,1) = Xpredict(k,1,:)*out_param(2,1) + out_param(1,1);
  laplaceDL(:,k,2) = Xpredict(k,2,:)*out_param(2,2) + out_param(1,2);
  laplaceDL(:,k,4) = Xpredict(k,3,:)*out_param(2,3) + out_param(1,3);
  laplaceDL(:,k,5) = Xpredict(k,4,:)*out_param(2,4) + out_param(1,4);
end 
% outputs
% velx_real, vely_real, velx_imag, vely_imag, xlayers, ylayers, trans, rotate, rotCent, scaling, sortIdx

xlayers = zeros(N,5,nv);
ylayers = zeros(N,5,nv);
for k = 1 : nv
  for il = 1 : 5
     Xl = o.destandardize(tracersX(:,il,k),trans(:,k),rotate(k),rotCent(:,k),scaling(k),sortIdx(:,k));
     xlayers(:,il,k) = Xl(1:end/2);
     ylayers(:,il,k) = Xl(end/2+1:end);
  end
end


% Do the far-field and correct near-field
NearV2V = vesicle.getZone([],1);    
% can be done only once as computeStokesInteractions also need that
zone = NearV2V.zone;

% First calculate the far-field
farField = zeros(2*N,nv);
for k = 1 : nv
  K = [(1:k-1) (k+1:nv)];
  farField(:,k) = o.exactLaplaceDL(vesicle, f, vesicle.X(:,k), K);
end
farField = farField(1:end/2,:); % scalar value

interpOp = []; 
for k = 1 : nv
 % layers around the vesicle j 
 Xin = [reshape(xlayers(:,:,k),1,5*N); reshape(ylayers(:,:,k),1,5*N)];
 velInput = reshape(laplaceDL(:,:,k), 1, 5*N); 
  
 interpOp{k} = rbfcreate(Xin,velInput,'RBFFunction','linear');
end

nearField = zeros(size(farField));

for k1 = 1 : nv
  K = [(1:k1-1) (k1+1:nv)];
  for k2 = K
    % points on vesicle k2 close to k1  
    J = find(zone{k1}(:,k2) == 1);  
    if numel(J) ~= 0
      % need tp subtract off contribution due to vesicle k1 since its layer
      % potential will be evaulated through interpolation
      potTar = o.exactLaplaceDL(vesicle, f, [xvesicle(J,k2);yvesicle(J,k2)],k1);
      nearField(J,k2) = nearField(J,k2) - potTar(1:numel(J)); % scalar value
         
      % now interpolate
      for i = 1 : numel(J)
        pointsIn = [xvesicle(J(i),k2);yvesicle(J(i),k2)];
        % interpolate for the k2th vesicle's points near to the k1th vesicle
        rbfVel = rbfinterp(pointsIn, interpOp{k1});
        nearField(J(i),k2) = nearField(J(i),k2) + rbfVel;
      end
    end % if numel(J)

  end % for k2
end % for k1

% finally add the corrected nearfield to the farfield
Fdlp = farField + nearField;


bufferVes = 2e-3;
% can't set buffer too big because near singular integration does not
% assign a value of 1 when near points cross.  This is because I did not
% swtich the direction of the normal for this case.  So, the lagrange
% interpolation points may look something like [0 1 1 1 1 ...] instead
% of [1 1 1 1 1 ...].  The local interpolant is affected by this and
% a value somewhere between 0 and 1 will be returned
icollisionVes = any(abs(Fdlp(:)) > bufferVes);


end % collisionDetection
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [laplaceDLPtar] = exactLaplaceDL(o,vesicle,f,Xtar,K1)
% pot = exactLaplaceDL(vesicle,f,Xtar,K1) computes the double-layer
% laplace potential due to f around all vesicles except itself.  Also
% can pass a set of target points Xtar and a collection of vesicles K1
% and the double-layer potential due to vesicles in K1 will be
% evaluated at Xtar.  Everything but Xtar is in the 2*N x nv format
% Xtar is in the 2*Ntar x ncol format

oc = o.oc;

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
