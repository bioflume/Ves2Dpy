%% In the MLARM_freeMany, one time step is taken as follows
% 1) Using previous time step's X and tension, compute bending and tension
% forces; then add them to get tracJump

% 2) Compute farFieldtracJump (single layer integral) due to tracJump
% Since throughout the time step vesicle configuration does not change, we
% need to predict the integral on the Fourier modes once so do the
% following:
[velx_real, vely_real, velx_imag, vely_imag, xlayers, ylayers, transNear, rotateNear, ...
    rotCentNear, scalingNear, sortIdxNear] = predictNearLayersOnceAllModes(X); 

% Then compute the farFieldtracJump with the tracJump
farFieldtracJump = o.computeStokesInteractionsNet(vesicle, tracJump, ...
    velx_real, vely_real, velx_imag, vely_imag, xlayers, ylayers, transNear, rotateNear, ...
    rotCentNear, scalingNear, sortIdxNear);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [velx_real, vely_real, velx_imag, vely_imag, xlayers, ylayers, trans, rotate, rotCent, scaling, sortIdx] = predictNearLayersOnceAllModes(X)
Nnet = numel(X(:,1))/2;
oc = curve;

in_param = o.torchNearInNorm;
out_param = o.torchNearOutNorm;

maxLayerDist = 1/Nnet; % be careful with this one, our new nets assume h (not sqrt(h))

nlayers = 3;
dlayer = (0:nlayers-1)'/(nlayers-1) * maxLayerDist;

% standardize input
Xstand = zeros(size(X));
nv = numel(X(1,:));
scaling = zeros(nv,1);
rotate = zeros(nv,1);
rotCent = zeros(2,nv);
trans = zeros(2,nv);
sortIdx = zeros(Nnet,nv);

tracersX = zeros(2*Nnet,3,nv);
for k = 1 : nv
  [Xstand(:,k),scaling(k),rotate(k),rotCent(:,k),trans(:,k),sortIdx(:,k)] = standardizationStep(X(:,k),Nnet);
  [~,tang] = oc.diffProp(Xstand(:,k));
  nx = tang(Nnet+1:2*Nnet);
  ny = -tang(1:Nnet);

  tracersX(:,1,k) = Xstand(:,k);
  for il = 2 : nlayers 
    tracersX(:,il,k) = [Xstand(1:end/2,k)+nx*dlayer(il); Xstand(end/2+1:end,k)+ny*dlayer(il)];
  end
end

% Normalize input
input_net = zeros(nv,2*Nnet,Nnet);

for ij = 1 : 128
for k = 1 : nv
  input_net(k,(ij-1)*2+1,:) = (Xstand(1:end/2,k)-in_param(1,1))/in_param(1,2);
  input_net(k,2*ij,:) = (Xstand(end/2+1:end,k)-in_param(1,3))/in_param(1,4);
end
end


modes = [(0:Nnet/2-1) (-Nnet/2:-1)];
modesInUse = 128;
modeList = find(abs(modes)<=modesInUse);


input_conv = py.numpy.array(input_net);
[Xpredict] = pyrunfile("near_vel_allModes_predict.py","output_list",input_shape=input_conv);

Xpredict = double(Xpredict);

for k = 1 : nv
velx_real{k} = zeros(Nnet,Nnet,3);
vely_real{k} = zeros(Nnet,Nnet,3);
velx_imag{k} = zeros(Nnet,Nnet,3);
vely_imag{k} = zeros(Nnet,Nnet,3);
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

xlayers = zeros(Nnet,3,nv);
ylayers = zeros(Nnet,3,nv);
for k = 1 : nv
  for il = 1 : 3
     Xl = destandardize(tracersX(:,il,k),trans(:,k),rotate(k),rotCent(:,k),scaling(k),sortIdx(:,k));
     xlayers(:,il,k) = Xl(1:end/2);
     ylayers(:,il,k) = Xl(end/2+1:end);
  end
end


end % predictNearLayersOnceAllModes

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function farField = computeStokesInteractionsNet(vesicle, tracJump, ...
        velx_real, vely_real, velx_imag, vely_imag, xlayers, ylayers, trans,...
        rotate,rotCent, scaling, sortIdx)



N = vesicle.N;
nv = vesicle.nv;
Ntar = N; Nsou = N;

NearV2T = vesicle.getZone([],1); 
zone = NearV2T.zone;
dist = NearV2T.dist;
nearest = NearV2T.nearest;
icp = NearV2T.icp;
argnear = NearV2T.argnear;
interpMat = lagrangeInterp;
interpOrder = size(interpMat,1);
p = ceil((interpOrder+1)/2);

% Predict velocity on layers
[velx, vely] = buildVelocityInNear(tracJump, velx_real, vely_real, velx_imag, vely_imag, trans, rotate, rotCent, scaling, sortIdx);

% First calculate the far-field
farField = zeros(2*N,nv);
vself = zeros(2*N,nv);
for k = 1 : nv
  K = [(1:k-1) (k+1:nv)];
  [~,farField(:,k)] = exactStokesSL(vesicle, tracJump, [], vesicle.X(:,k), K);
  vself(:,k) = [velx(:,1,k);vely(:,1,k)]; % velocity on vesicle itself  
end

% THE ALGORITHM BELOW IS CLASSICAL HEDGEHOG - I PUT HERE SO THAT YOU CAN
% ALSO SEE HOW THE SUBTRACTION OF WRONG NEAR-FIELD IS DONE FIRST
% YOU WILL USE KNN FOR THIS STEP TO FIND NEAR POINTS
% THEN USE RBF FOR INTERPOLATION

hves = vesicle.length/vesicle.N;
nearField = zeros(size(farField));
beta = 1.1;
% small buffer to make sure Lagrange interpolation points are
% not in the near zone

for k1 = 1:nv
  
  K = [(1:k1-1) (k1+1:nvTar)];
  for k2 = K
    % set of points on vesicle k2 close to vesicle k1
    J = find(zone{k1}(:,k2) == 1);
    
    if (numel(J) ~= 0)
      % closest point on vesicle k1 to each point on vesicle k2 
      % that is close to vesicle k1
      indcp = icp{k1}(J,k2);
      
      for j = 1:numel(J)
        % index of points to the left and right of the closest point  
        pn = mod((indcp(j)-p+1:indcp(j)-p+interpOrder)' - 1,Nsou) + 1;
        
        % x-component of the velocity at the closest point
        v = filter(1,[1 -full(argnear{k1}(J(j),k2))],...
          interpMat*vself(pn,k1));
        vel(J(j),k2,k1) = v(end);  
        
        % y-component of the velocity at the closest point
        v = filter(1,[1 -full(argnear{k1}(J(j),k2))],...
          interpMat*vself(pn+Nsou,k1));
        vel(J(j)+Ntar,k2,k1) = v(end);
        
      end

      % HERE IS THE SUBTRACTION
%     compute values of velocity at required intermediate points
%     using local interpolant
      [~,potTar] = op.exactStokesSL(vesicle,tracJump,[],...
           [vesicle.X(J,k2);vesicle.X(J+Ntar,k2)],k1);
    
      % Need to subtract off contribution due to vesicle k1 since its
      % layer potential will be evaulted using Lagrange interpolant of
      % nearby points
      nearField(J,k2) =  nearField(J,k2) - potTar(1:numel(J));
      nearField(J+Ntar,k2) =  nearField(J+Ntar,k2) - potTar(numel(J)+1:end);
      
      XLag = zeros(2*numel(J),interpOrder - 1);
      % initialize space for initial tracer locations
      % Lagrange interpolation points coming off of vesicle k1 All
      % points are behind Xtar(J(i),k2) and are sufficiently far from
      % vesicle k1 so that the Nup-trapezoid rule gives sufficient
      % accuracy
      for i = 1:numel(J)
        nx = (vesicle.X(J(i),k2) - nearest{k1}(J(i),k2))/...
            dist{k1}(J(i),k2);
        ny = (vesicle.X(J(i)+Ntar,k2) - nearest{k1}(J(i)+Ntar,k2))/...
            dist{k1}(J(i),k2);
        XLag(i,:) = nearest{k1}(J(i),k2) + ...
            beta*hves*nx*(1:interpOrder-1);
        XLag(i+numel(J),:) = nearest{k1}(J(i)+Ntar,k2) + ...
            beta*hves*ny*(1:interpOrder-1);
      end
      % evaluate velocity at the lagrange interpolation points
      [~,lagrangePts] = op.exactStokesSL(vesicle,tracJump,[],XLag,k1);

      
      for i = 1:numel(J)
        % Build polynomial interpolant along the one-dimensional
        % points coming out of the vesicle
        Px = interpMat*[vel(J(i),k2,k1) ...
            lagrangePts(i,:)]';
        Py = interpMat*[vel(J(i)+Ntar,k2,k1) ...
            lagrangePts(i+numel(J),:)]';
        
        % Point where interpolant needs to be evaluated
        dscaled = full(dist{k1}(J(i),k2)/(beta*hves*(interpOrder-1)));
        

        % Assign higher-order results coming from Lagrange 
        % integration to velocity at near point.  Filter is faster
        % than polyval
        v = filter(1,[1 -dscaled],Px);
        nearField(J(i),k2) = nearField(J(i),k2) + ...
            v(end);
        v = filter(1,[1 -dscaled],Py);
        nearField(J(i)+Ntar,k2) = nearField(J(i)+Ntar,k2) + ...
            v(end);
        
      end % i
    end % numel(J) ~= 0
    % Evaluate layer potential at Lagrange interpolation
    % points if there are any
  end % k2
end % k1
% farField


% finally add the corrected nearfield to the farfield
farField = farField + nearField;


end % computeStokesInteractionsNet

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [velx, vely] = buildVelocityInNear(tracJump, velx_real, vely_real, velx_imag, vely_imag, trans, rotate, rotCent, scaling, sortIdx)
nv = numel(rotate);
Nnet = numel(sortIdx(:,1));

% standardize tracJump
fstandRe = zeros(Nnet, nv);
fstandIm = zeros(Nnet, nv);
for k = 1 : nv
  fstand = standardize(tracJump(:,k),[0;0], rotate(k), [0;0], 1, sortIdx(:,k));
  z = fstand(1:end/2) + 1i*fstand(end/2+1:end);
  zh = fft(z);
  fstandRe(:,k) = real(zh); 
  fstandIm(:,k) = imag(zh);
end


velx = zeros(Nnet,3,nv); 
vely = zeros(Nnet,3,nv);
for k = 1 : nv
  velx_stand = zeros(Nnet,3);
  vely_stand = zeros(Nnet,3);
  for il = 1 : 3
     velx_stand(:,il) = velx_real{k}(:,:,il) * fstandRe(:,k) + velx_imag{k}(:,:,il)*fstandIm(:,k);
     vely_stand(:,il) = vely_real{k}(:,:,il) * fstandRe(:,k) + vely_imag{k}(:,:,il)*fstandIm(:,k);
     
     vx = zeros(Nnet,1);
     vy = zeros(Nnet,1);
     
     vx(sortIdx(:,k)) = velx_stand(:,il);
     vy(sortIdx(:,k)) = vely_stand(:,il);

     VelBefRot = [vx;vy];
     
     VelRot = o.rotationOperator(VelBefRot, -rotate(k), [0;0]);
     velx(:,il,k) = VelRot(1:end/2); vely(:,il,k) = VelRot(end/2+1:end);
  end
end


end