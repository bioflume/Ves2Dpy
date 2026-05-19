function flag = rayCasting(y, S)
    p0   = 1.1 * max(S) * ones(2, 1);           % a point outside S
    ray  = [p0 y];
    % V1 and V2 are unchanging, defined from ray
    V1 = ray(:, 1); V2 = ray(:, 2);
    % V3 is 2x1xN so each pair of neighbours from S is its own "page"
    V3 = reshape(S,2,1,[]);
    % V4 is a wrap-around of V3, move all pairs by one page
    V4 = V3(:,:,[2:end,1]);
    % Compute B in one hit, no loop required now we have paged V1 and V3
    B = V3 - V1;

    % Right-hand side of A is defined as diff of V3 and V4 as before
    AR = V3 - V4;
    % Left-hand side of A was always fixed by ray anyway, quick repmat
    AL = repmat(V2-V1, 1, 1, size(AR,3));
    % Define A, 2x2xN square matrices "paged" for each check
    % where N = 0.5*size(S,1)
    A = [AL, AR];

    % check for ill-conditioned matrices from the abs determinant
    % faster to compute the det manually as it's only 2x2 and we can
    % vectorise it this way
    idx = abs( A(1,1,:).*A(2,2,:) - A(1,2,:).*A(2,1,:) ) >= 1e-7;
    % For all well conditioned indices, combute A\b
    alpha = squeeze( pagemldivide( A(:,:,idx), B(:,:,idx) ) );
    % Count the number of elements where both rows of alpha are 0<=a<=1
    hit = nnz( (0 <= alpha(1,:)) & (alpha(1,:) <= 1) & (0 <= alpha(2,:)) & (alpha(2,:) <= 1) );   

    % Output flag
    if mod(hit, 2) == 0
        flag = 0;
    else
        flag = 1;
    end
end