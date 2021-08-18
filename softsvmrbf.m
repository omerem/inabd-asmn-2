%% Copyright (C) 2019 lior_
%% Author: lior_ <lior_@DESKTOP-RAG7FMP>
%% Created: 2019-12-28



function alpha = softsvmrbf (lambda, sigma, m, d, Xtrain, Ytrain)
  %% x = [alpha_1, ...., alpha_m, beta_1,...., beta_m] size 2m X 1
  
  %% G_i,j = K(x_i, X_j)
  G = zeros(m);
  for i = 1:m
    for j = i:m
        if i ~= j
            G(j,i) = rbf_kernel(Xtrain(i, :), Xtrain(j, :), sigma); 
            G(i,j) = G(j,i);    %% G simetric
        else
            G(i,j) = rbf_kernel(Xtrain(i, :), Xtrain(j, :), sigma);
        end
    end
  end
		
  H = [2*lambda*G, zeros(m) ; zeros(m, 2*m)]; %%from target function
  A = [G .* -Ytrain, -eye(m)];  %%constraint: -y_i<alpha, G[i]> - beta_i <= -1
  A = [A ; zeros(m), -eye(m)];  %%constrain: -beta_i <= 0
  b = [-ones(m, 1) ; zeros(m, 1)];
  f = ([zeros(1, m), ones(1, m)])' / m;
  
  %% before use: pkg load optim
  x = quadprog(H, f, A, b);  %%solve program
  
  alpha = x(1:m);
end


function output = rbf_kernel (x1, x2, sigma)
  
  v = x1 - x2;
  power = -(v * v') / (2*sigma);
  
  output = exp(power);
end
