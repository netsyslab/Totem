%% SCRIPT TO GENERATE EXAMPLES FOR "COMMUNITY STRUCTURE IN GRAPHS"
%Distributed as part of the BTER Project, Sandia National Labs, 2011.
%For more information, contact Tamara G. Kolda, tgkolda@sandia.gov.

%% Check for MATLAB_BGL
% This script requires the MATLAB_BGL toolbox.
if ~exist('clustering_coefficients.m','file')
    error('Must install MATLAB_BGL toolbox');
end

%% ca-AstroPh
graph_name = 'ca-AstroPh';
load(graph_name)

G = preprocess_graph(G);

ddist = degree_dist(G);
C = bter(ddist, 'rho_init', 0, 'edge_inc', 0.05);
B = bter(ddist, 'rho_init', .95, 'rho_decay', 0.5, 'last_empty', true, 'd1d1_edges', 0);

compare_graphs(G, B, C, graph_name, true);

%% cit-HepPh
graph_name = 'cit-HepPh';
load(graph_name)

G = preprocess_graph(G);

ddist = degree_dist(G);
C = bter(ddist, 'rho_init', 0, 'edge_inc', 0.05);
rhofunc = @(d,dmax)  + 0.7*(1 - 0.6*(log(d-1)/log(dmax-1))^3);
B = bter(ddist, 'rho_func', rhofunc, 'last_empty', true, 'd1d1_edges', 0);

compare_graphs(G, B, C, graph_name, true);

%% soc-Epinions1
graph_name = 'soc-Epinions1';
load(graph_name)

G = preprocess_graph(G);

ddist = degree_dist(G);
C = bter(ddist, 'rho_init', 0, 'edge_inc', 0.05);
B = bter(ddist, 'rho_init', .7, 'rho_decay', 1.25, 'last_empty', true, 'd1d1_edges', 0);

compare_graphs(G, B, C, graph_name, true);

%% ca-CondMat
graph_name = 'ca-CondMat';
load(graph_name)

G = preprocess_graph(G);

ddist = degree_dist(G);
C = bter(ddist, 'rho_init', 0, 'edge_inc', 0.05);
B = bter(ddist, 'rho_init', .95, 'rho_decay', 0.95, 'last_empty', true, 'd1d1_edges', 0);

compare_graphs(G, B, C, graph_name, true);
