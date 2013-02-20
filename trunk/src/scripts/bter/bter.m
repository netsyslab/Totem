function G = bter(ddist,varargin)
%BTER Block Two-Level Erdos-Renyi Graph Generation
%
%   G = BTER(D) generates a block two-level Erdos-Renyi (BTER) graph with
%   the specified degree distribution, i.e., D(i) = the number of nodes
%   with degree i. The BTER model has two phases. In Phase 1, we create
%   small independent communities comprising nodes with degree 2 or higher
%   and such that all nodes in a community are approximately the same
%   degree. Each small community is modeled as an Erdos-Renyi graph where
%   the link probability is a function that depends on the least degree in
%   the community. After Phase 1, each node has some "leftover" degree. In
%   Phase 2, we connect the independent communities as well as the degree-1
%   nodes by using a Chung-Lu model (also known as weighted Erdos-Renyi)
%   based on the "leftover" degrees.
%
%   DETAILS
%
%   In Phase 1, the default connectivity of the small communities is
%   denoted by rho and computed as: 
%    
%      rho = rho_init * ( 1 - rho_decay * log(d-1)/log(dmax-1) )^2 )  (*)
%
%   where d is the smallest degree in the community and dmax is the largest
%   possible degree. The parameters rho_init and rho_decay are user
%   specified parameters that default to 0.9 and 0.5, respectively. It is
%   also possible to use any user-specified function that takes d and dmax
%   as inputs.
%
%   In Phase 2, the Chung-Lu model operates on the "leftover" expected
%   degrees after the communities have been formed. In the basic Chung-Lu
%   model, each edge is generated by selecting its endpoints proportional
%   to e(i)/sum(e) where e(i) is the expected (leftover) degree for node i.
%   Duplicate edges and self-links are discarded, so, by default, we
%   generate 10% extra edges to make up for those that are discarded. 
%
%   We have also observed that many of the degree-one nodes (completely
%   ignored in Phase 1) become isolated in a naive implementation of the
%   Chung-Lu model. Therefore, we "manually" create edges for some
%   proportion of the degree-one nodes (the default proportion is 75%). Of
%   those manual links, some links will be to other degree-one nodes while
%   the remainder go to nodes of degree 2 or higher. By default, if p is
%   the number of degree-one nodes that are handled manually and M is the
%   total number of edges in the graph (both phases), the we set the number
%   of degree-one-to-degree-one edges to be round(p^2/(2*M)). Of the
%   degree-one nodes that are not handled "manually", we have found that
%   increasing their expected degree yields a more accurate degree
%   distribution; by default, that tweak is 0.1 (which sets the expected
%   degree to 1.1). 
%
%   USER-SPECIFIED PARAMETERS
%
%   G = BTER(D,'param',value,...) sets various options as follows.
%
%   - 'rho_func'   - Equation to determine the connectivity of the small
%                    communities in Phase 1. Default is Equation (*) above.
%                    The function should take two paramters: d (the
%                    smallest degree in the community) and dmax (the
%                    maximum degree in the entire graph). Examples are
%                    given below.
%   - 'rho_init'   - Parameter in Equation (*) above. Default is 0.9.
%                    Ignored if 'rho_func' is specified.
%   - 'rho_decay'  - Parameter in Equation (*) above. Default is 0.5.                    
%                    Ignored if 'rho_func' is specified.
%   - 'last_empty' - Set the last community's connectivity to zero. This is
%                    typically comprised of just a few high-degree nodes.
%                    Default is false.
%   - 'edge_inc'   - Amount that Phase 2 edges should be increased to
%                    account for edge duplication. Default is 0.1 (10%).
%   - 'd1_manual'  - Proportion of degree-one nodes to handle manually. The
%                    remainder are put into the general Phase 2 procedure.
%                    Default is 0.75 (75%).
%   - 'd1d1_edges' - Number of isolated edges between degree 1 nodes that
%                    are handled manually.
%   - 'd1_inc'     - In Phase 2, for those degree-one nodes that are not
%                    handled manually, the amount that their expected
%                    degree should be increated. Default is 0.1 (increasing
%                    the expected degree to 1.1).
%   - 'verbose'    - Print information. Default is true.
%   - 'debug'      - Extra error checking is turned on. Default is false.
%
%   Abdullah: I changed the function to return only the final graph "G". Doing
%             this allowed to generate larger graphs by getting rid of some 
%             out-of-memory errors. The following is the original signature of
%             the function:
%   [G, G1, G2, csz, crho, cid, d] = BTER(D,...) returns additional
%   information as follows:
%      G1 = Graph from Phase 1
%      G2 = Graph from Phase 2
%      csz = Community sizes; csz(c) = number of nodes in community c.
%      crho = Community connectivity; crho(c) = rho for community c.
%      cid = Community ids; cid(i) = community id for node i.
%      d = Expected degree of each node; d(i) = expected degree of node i.
%
%   Note that the number of communities is equal to length(csz). Also, the
%   expected degrees in d yield the desired degree distribution in the
%   input D.
%      
%   EXAMPLES
%
%   % Generate BTER graph with power-law degree distribution
%   gamma = 1.9; maxdeg = 150;
%   ddist = round(maxdeg^(gamma)./(1:maxdeg).^(gamma))';

% Lauro: Quick note on this example based on the discussion between Lauro
% and Elizeu via email on 20/Feb/2012. 
% The Zipf form of a power-law is:
% $y = a*r^(-\alpha)$, where, in our context, $y$ will be the number of
% nodes that have degree $r$, and $a$ is a location parameter.
% So, it seems $a = maxdeg^(gamma)$, which is a constant, and you remove
% the sign from $\alpha$ by making $1/(r^\alpha)$.
% Note that we can generate the distribution by using $\a$ different
% of $maxdeg^(gamma)$ if this is convinient. 
%
%   G = bter(ddist, 'rho_init', 0.99, 'rho_decay', 0.8);
%
%   % Use a custom rho function
%   G = bter(ddist, 'rho_func', @(d,dmax) (dmax - d + 2)/dmax);
%
%   % Basic Chung-Lu
%   G = bter(ddist, 'rho_init', 0, 'd1_manual', 0, 'd1_inc', 0);
%   
%   CITATION
%
%   Community structure, triangles, and scale-free collections of
%   Erdos-Renyi graphs by C. Seshadri, T. G. Kolda, and A. Pinar, 2011.
%
%Distributed as part of the BTER Project, Sandia National Labs, 2011.
%For more information, contact Tamara G. Kolda, tgkolda@sandia.gov.


%% Get inputs
if nargin < 1
    error('At least one input is required');
end

%% Process remaining inputs
% Lauro: Octave does not support inputParser yet. So, I changed this part of
% the code to make most of the parameters fixed values. The values used are
% based on the default values of the original implementation.

% params = inputParser;
% params.addParamValue('rho_init', 0.9);
% params.addParamValue('rho_decay', 0.5);
% params.addParamValue('rho_func', []);
% params.addParamValue('d1_manual', 0.75);
% params.addParamValue('d1_inc', 0.1);
% params.addParamValue('d1d1_edges', -1);
% params.addParamValue('edge_inc', 0.1);
% params.addParamValue('verbose',true,@islogical);
% params.addParamValue('debug',false,@islogical);
% params.addParamValue('last_empty',false,@islogical);
% params.parse(varargin{:});
% 
% rho_init = params.Results.rho_init;
% rho_decay = params.Results.rho_decay;
% rho_func = params.Results.rho_func;
% if isempty(rho_func)
%     rho_func = @(d,dmax) rho_init * (1 - rho_decay *((log(d-1))/(log(dmax)))^2);
% end
% 
% edge_inc = params.Results.edge_inc;
% d1_manual = params.Results.d1_manual;
% d1d1_edges = params.Results.d1d1_edges;
% d1_inc = params.Results.d1_inc;
% 
% last_empty = params.Results.last_empty;
% 
% debug = params.Results.debug;
% verbose = params.Results.verbose;

rho_init = 0.9;
rho_decay = 0.5;
rho_func = [];
d1_manual = 0.75;
d1_inc = 0.1;
d1d1_edges = -1;
edge_inc = 0.1;
verbose = true;
debug = false;
last_empty = false;

rho_func = @(d,dmax) rho_init * (1 - rho_decay *((log(d-1))/(log(dmax)))^2);

%% Set some parameters

% Number of nodes
nnodes = sum(ddist);

% Max degree
maxdeg = length(ddist);

% Number of edges 
nedges = dot((1:maxdeg)',ddist)

if verbose
    fprintf('Constructing BTER graph with %d nodes and %d edges.\n', nnodes, nedges);
end

%% Set some more parameters

% Number of degree 1 nodes
p = ddist(1);


%% Call BTER-Phase 1
if verbose
    fprintf('Running BTER Phase 1...\n');
end

% --- Bookkeeping on the blocks ---
% Abdullah: the following two arrays are not used in generating the graph
% therefore, I am commenting them out to reduce memory footprint
# maxblk = nnodes;
# blksize = zeros(maxblk,1);
# blkrho = zeros(maxblk,1);
bloc = 1;

% --- Degree Lists ---
deglist = zeros(nnodes,1);
deglist2 = zeros(nnodes,1);
% Abdullah: the following array is not used in generating the graph,
% therefore I am commenting it out to reduce memory footprint
% blklist = zeros(nnodes,1);

% --- Handle degree 1 nodes ---
deglist(1:p) = 1;
deglist2(1:p) = 1;
nloc = p + 1;
ddist(1) = 0;

% source/destination pairs for Phase 1
edges = zeros(nedges,2, 'int32');
eloc = 1;

if verbose
    fprintf('Entering Phase 1 loop...\n');
end

% Phase 1 Loop
d = 2;
while d <= maxdeg

    % Check if every node has been assigned
    if sum(ddist) == 0
        break;
    end
    
    % Pick smallest d such that ddist > 0
    while (ddist(d) == 0)
        d = d + 1;
        if verbose
            fprintf("Phase 1 processing degree %d\n", d);
        end
    end   
       
    % Fill up the pattern
    psize = min(d+1,sum(ddist));
    pattern = zeros(psize,1);
    ploc = 1;
    dd = d;
    while ploc <= psize
        n = min(ddist(dd), psize-ploc+1);
        ploc2 = ploc+n-1;
        pattern(ploc:ploc2) = dd;
        ploc = ploc2 + 1;
        ddist(dd) = ddist(dd) - n;
        dd = dd + 1;
    end
    
    % Determine the ER probability
    rho = rho_func(d,maxdeg);

    % Make the last block empty
    if (sum(ddist) == 0) && (last_empty)
        rho = 0;
    end
    
    % Update deglist and deglist2
    nloc2 = nloc + psize - 1;
    deglist(nloc:nloc2) = pattern;
    deglist2(nloc:nloc2) = pattern - rho*(psize-1);
    
    % Update block info
    % Abdullah: the following arrays are not used in generating the graph
    % therefore, I am commenting them out to reduce memory footprint
    # blklist(nloc:nloc2) = bloc;
    # blksize(bloc) = psize;
    # blkrho(bloc) = rho;
    bloc = bloc+1;
    
    % Create ER graph and extract and save edges
    Gtmp = ergraph_prob_dense(psize,rho);
    [ii,jj] = find(Gtmp);
    ne = nnz(Gtmp);
    if ne > 0
        eloc2 = eloc + ne - 1;
        edges(eloc:eloc2,:) = [ii+nloc-1 jj+nloc-1];
        eloc = eloc2 + 1;
    end
    
    % Update node location
    nloc = nloc2 + 1;
end

% Resize the arrays
% Abdullah: the following arrays are not used in generating the graph
% therefore, I am commenting them out to reduce memory footprint
# blksize = blksize(1:bloc-1);
# blkrho = blkrho(1:bloc-1);
edges = edges(1:eloc-1,:);

% Create G1
G1 = sparse(edges(:,1), edges(:,2), 1, nnodes, nnodes);

if debug 
    check(G1,'G1'); 
end

if verbose
    % Abdullah: the following printf statements access the previously 
    % commented-out blkrho array, hence they are also commented-out
    # fprintf('Phase 1 Max Connectivity: %d%%\n', round(100*max(blkrho)));
    # fprintf('Phase 1 Min Connectivity: %d%%\n', round(100*min(blkrho)));
    fprintf('Phase 1 communities: %d\n', bloc-1);
    fprintf('Phase 1 edges: %d\n',nnz(G1));
end

%% Prep for BTER-Phase 2
if verbose
    fprintf('Running BTER Phase 2...\n');
end

% Divvy up degree-one edges into those that will be handled "manually" and
% those that will be handled in the Chung-Lu model.
newp = min(p, round(d1_manual * p));
deglist2(1:newp) = 0;
deglist2(newp+1:p) = 1 + d1_inc;

if verbose
    fprintf('Phase 2 "Manual" Degree 1 Nodes: %d\n', newp);
    fprintf('Phase 2 Chung-Lu Degree 1 Nodes: %d\n', p-newp);
    fprintf('Phase 2 Chung-Lu Degree 1 Nodes Boost: %d%%\n', round(100*d1_inc));
    fprintf('Phase 2 Chung-Lu Percent Extra Edges: %d%%\n', round(edge_inc*100));
end

p = newp;

%% Call BTER-Phase 2a: Degree 1 to Degree 1 connections
if d1d1_edges == -1
    q = 2*round(p^2/(2*sum(deglist)));
else
    q = 2*d1d1_edges;
end

q = min(q, p);

if q == 0
    G2a = sparse(nnodes,nnodes);
else
    TMP = repmat([0 1; 1 0], q, 1);
    G2a = spdiags(TMP, [-1 1], q, q);
    [ii jj] = find(G2a);
    G2a = sparse(ii,jj,1,nnodes,nnodes);
end

if debug
    check(G2a,'G2a'); 
end

if verbose
    fprintf('Phase 2a (Manual Degree 1-Degree 1) edges: %d\n', nnz(G2a));
end

%% BTER-Phase 2b: Degree 1 to Degree 2+ connections
G2b = ergraph_one_sparse(deglist2, q+1, p);

if debug
    check(G2b,'G2b'); 
end;

if verbose
    fprintf('Phase 2b (Manual Degree 1-Degree 2+) edges: %d\n', nnz(G2b));
end

%% Call BTER-Phase 2c
% Calculation proportion of edges remaining after completion of phase 2b
foo = 1 + edge_inc - (2*(p-q) / (sum(deglist2)+(p-q)));
deglist2c = max(0,foo*deglist2);
G2c = ergraph_prob_sparse(deglist2c);

if debug
    check(G2c,'G2c'); 
end;

if verbose
    fprintf('Phase 2c (Chung-Lu) edges: %d\n', nnz(G2c));
end

%% Assemble G2
G2 = G2a + G2b + G2c;

if debug
    check(G2,'G2'); 
end;

if verbose
    fprintf('Phase 2 total edges: %d\n', nnz(G2));
end


%% Assemble Final Matrix
G = spones(G1+G2);

if debug
    check(G,'G'); 
end;  

d_final = full(sum(G,2));
nnodes_final = sum(d_final>0);
if verbose
    fprintf('Final number of non-isolated nodes: %d (%d%% of desired)\n', nnodes_final, round(100*nnodes_final/nnodes));
    fprintf('Final numbder of edges: %d (%d%% of desired)\n', nnz(G), round(100*nnz(G)/nedges));
end


function G = ergraph_prob_dense(nnodes, rho)
%ERGRAPH_PROB_DENSE Create symmatric Erdos-Renyi graph

% This is the classical way of generating ER graphs, especially in the
% case where are links are equally likely.
T = rho * ones(nnodes);
R = rand(nnodes);
G = double(R < T);

% Make symmetric and remove self-links
G = triu(G,1);
G = G + G';
G = double(G > 0);

function G = ergraph_prob_sparse(deglist)
%ERGRAPH_PROB_SPARSE Chung-Lu model for a given desired degree list    

% Preliminary computations
nnodes = length(deglist);
nedges = sum(deglist);
prob = deglist/nedges;

% Create bins
bins = min([0 cumsum(prob')],1);
bins(end) = 1;

% Create indices
nedges = round(nedges/2);
% Lauro: I understand the usage of ~ means a void assigment. 
% Octave does not support ~ and ~ is not used later. So, I commented it.
% [~, ii] = histc(rand(nedges,1),bins);
% [~, jj] = histc(rand(nedges,1),bins);
[garbage, ii] = histc(rand(nedges,1),bins);
[garbage, jj] = histc(rand(nedges,1),bins);

% Create symmetric graph
G = spones(sparse([ii;jj],[jj;ii],1,nnodes,nnodes));

% Remove self-links
G = spdiags(zeros(nnodes,1),0,G);

function G = ergraph_one_sparse(deglist, p1, p2)
%ERGRAPH_ONE_SPARSE Randomly generate edges attached to degree-one nodes

% Preliminary computations
nnodes = length(deglist);
prob = deglist/sum(deglist);

% Error check (implies no possibility of self links)
if any(deglist(p1:p2)>0)
    error('Error in manual handling of degree-one nodes')
end

% Create bins
bins = min([0 cumsum(prob')],1);
bins(end) = 1;

% Create indices
nedges = p2-p1+1;
ii = (p1:p2)';
% Lauro: I understand the usage of ~ means a void assigment. 
% Octave does not support ~ and ~ is not used later. So, I commented it.
% [~, jj] = histc(rand(nedges,1),bins);
[garbage, jj] = histc(rand(nedges,1),bins);

% Create symmetric graph
G = spones(sparse([ii;jj],[jj;ii],1,nnodes,nnodes));


function check(G, name)
%CHECK Check that graph is simple and has no self-links   
if ~isequal(G,G')
    warning('%s is not symmetric', name);
end

for i = 1:size(G,1)
    if G(i,i) > 0
        warning('%s diagonal entry (%d,%d) is nonzero',name,i,i);
    end
end

idx = G > 1;
if nnz(idx) > 0
    warning('%s has some non-binary entries',name);
end
    
