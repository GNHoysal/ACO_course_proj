function ACO_proj_script()
    alpha = 1;
    %gamma = 1;
        
    [A, b, W] = create_A_b_W(); %creating entities A, b, and w
    
    %siz = size(W);
    
    I = eye(200, 200);
    
    [gamma_star_I, beta_I, sigma_I] = get_gamma_star(A); % computing gamma* for A
    
    E = find_optimal_metric(I, (1/beta_I)*I, (1/sigma_I)*I); %computing preconditioning matrix
    
    gamma_space_I = gamma_star_I*logspace(-3, 2, 10); % computing gamma* for preconditioned case
    
    [gamma_star_E, beta_E, sigma_E] = get_gamma_star(A*E);
    
    gamma_space_E = gamma_star_E*logspace(-3, 2, 10);
    
    lc_I = zeros(10);
    
    lc_E = zeros(10);
    
    fprintf('%3s\t%10s\t%10s\n', 'gamma', ...
      'iter', 'rel_tol');
    
    for q=[1:10]    
        lc_I(q) = ADMM(A, b, W, alpha, gamma_space_I(q), I); %compute number of iterations taken for said relative accuracy - non-preconditioned case
        lc_E(q) = ADMM(A, b, W, alpha, gamma_space_E(q), I); %compute number of iterations taken for said relative accuracy - preconditioned case
    end
    
    %plotting the results
    lg_I = loglog(gamma_space_I, lc_I, '-r');
    hold on;
    lg_E = loglog(gamma_space_E, lc_E, '-b');
    grid on;
    %lg(1).Color = [0 1 0];
    %lg(2).Color = [1 0 0];
    title('\gamma vs iterations');
    xlabel('\gamma values');
    ylabel('number of iterations');
    
    legend('non-preconditioned', 'preconditioned');
    shg
end

function [g_star, beta, sigma] = get_gamma_star(A) % function to get gamma* of A
    e = eig(A'*A);
    sigma = min(e);
    if sigma <= 0
        a;
    end
    
    beta = norm(A);
    
    g_star = 1/sqrt(sigma*beta);
end

function [A, b, W] = create_A_b_W() % function to create random entities A, b, and w
    A = zeros(300, 200);
    b = zeros(300, 1);
    %W = zeros(20, 20);
    
    for j=[1:200]
       for i=[1:300]
          r = unifrnd(0, 1);
          if r<=0.1
             A(i,j) = normrnd(0, 1);
          end          
       end  
    end
    
    for i=[1:30]
        b(i, 1) = normrnd(0, 1);
    end
    
    W = unifrnd(0, 1);            
end


function loop_counter = ADMM(A, b, W, alpha, gamma, mat) % ADMM algorithm
    %intializing variables
    temp = size(A);
    x_new = zeros(temp(1,2), 1);
    y_new = zeros(temp(1,2), 1);
    u_new = zeros(temp(1,2), 1);
    z_new = zeros(temp(1,2), 1);
    
    loop_counter = 0; % to track number of iterations
    
    while 1
        x = x_new;
        y = y_new;
        u = u_new;
        z = z_new;
        
        [x_new , y_new, u_new, z_new] = ADMM_update(x, y, u, A, b, W, alpha, gamma, mat); % one update step of ADMM
        
        loop_counter = loop_counter + 1; % update iteration count
        
         fprintf('%3f\t%10d\t%10f\n', gamma, ...
      loop_counter, norm(z_new - z));
        
        if norm(z_new - z) <= 10^(-4) % break if relative accuracy is less than specified
           break;
        end
    end
    
end

function [x, y, u, z] = ADMM_update(x_old, y_old, u_old, A, b, W, alpha, gamma, E)
    
    
    x1 = inv(A'*A + (2*gamma)*eye(size(A'*A))) * (A'*b + (2*gamma)*(E*y_old - u_old) ) ; %update x_k+1
    
    x_temp = (2*alpha*E*x1) - (1 - (2*alpha))*(-E*y_old); % update x^A_k+1
      
    b_temp = x_temp + u_old;
    
    y1 = max( 0, b_temp - (W/gamma) ) - max( 0, -b_temp - (W/gamma) ); % update y_k+1 (inspired from an implementation on ADMM with lasso minimization: https://web.stanford.edu/~boyd/papers/admm/lasso/lasso.html)
    u = u_old + (x_temp - E*y1); % update u_k+1
    z = gamma*(u+E*y1); % update u_k+1
    x = x1;
    y = y1;

end


function E = find_optimal_metric(A, H, L) % function to find optimal metric
    
    Q = A*inv(H)*A';
    s = size(Q);
    s1 = s(1,1);
    I = eye(s1, s1);
    
    cvx_begin quiet
        variable t;
        variable M(s1,s1) diagonal;
        
        minimize t
        subject to
            M - Q == semidefinite(s1, s1);
            t*Q - M == semidefinite(s1, s1);         
    cvx_end
    
%     cvx_begin quiet
%         variable t;
%         variable M(s1,s1) diagonal;
%         
%         minimize -t
%         subject to
%             I - R*M*R' == semidefinite(s1, s1);
%             R*M*R' - t*I == semidefinite(s1, s1);         
%     cvx_end
    
    E = sqrtm(inv(M));

end
