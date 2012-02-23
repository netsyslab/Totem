function compare_graphs(G, B, C, graph_name, print_figs)
%COMPARE_GRAPHS Compare original graph with BTER and CL models.
%
%   COMPARE_GRAPHS(G,B,C,NAME) compares an original graph, G, with its
%   BTER and CL models, B and C, respectively. The graph name is NAME. 
%
%   COMPARE_GRAPHS(G,B,C,NAME,true) also saves the figures to files using
%   NAME as the file prefix.
%
%Distributed as part of the BTER Project, Sandia National Labs, 2011.
%For more information, contact Tamara G. Kolda, tgkolda@sandia.gov.

%% Set-up
graphs = {G,B,C};
names = {graph_name, 'BTER', 'CL'};
maxdegs = cellfun(@(x) max(full(sum(x,2))), graphs);
maxdeg = max(maxdegs);
neigs = 25;

%% Printing options
symbol = {'rd', 'b*', 'g+'};
fontsize = 20;
markersize = 12;
linewidth = 1.25;
width = 6;
height = 5;
if ~exist('print_figs','var')
    print_figs = false;
end


%% Degree Distribution
figure(1); clf;
pos = get(gcf,'Position');
set(gcf,'Position',[pos(1) pos(2) width*100, height*100]);
for i = 1:3
    ddist = degree_dist(graphs{i});
    ddist(end+1:maxdeg) = 0;
    loglog(1:maxdeg, ddist, symbol{i}, 'MarkerSize', markersize, 'LineWidth', linewidth);
    hold on;
end
hold off;
set(gca, 'FontSize', fontsize, 'LineWidth', linewidth);
legend(names, 'LineWidth', linewidth);
title('Degree Distribution');
xlabel('Degree');
ylabel('Count');

if print_figs
    print_figure(gcf, width, height, [graph_name '-dd']);
end

%% Clustering Coefficient
figure(2); clf;
pos = get(gcf,'Position');
set(gcf,'Position',[pos(1) pos(2) width*100, height*100]);
for i = 1:3
    [ccpd,gcc{i}] = clustering_measures(graphs{i});
    ccpd(end+1:maxdeg) = 0;
    semilogx(2:maxdeg, ccpd(2:maxdeg), symbol{i}, 'MarkerSize', markersize, 'LineWidth', linewidth);
    hold on;
end
hold off;
set(gca, 'FontSize', fontsize, 'LineWidth', linewidth);
legend(names, 'LineWidth', linewidth);
title('Clustering Coefficient');
xlabel('Degree');
ylabel('Avg. Clustering Coefficient');

if print_figs
    print_figure(gcf, width, height, [graph_name '-cc']);
end

%% Scree Plot
figure(3); clf;
pos = get(gcf,'Position');
set(gcf,'Position',[pos(1) pos(2) width*100, height*100]);
for i = 1:3
    evals = eigs(graphs{i},neigs);
    plot(1:neigs, abs(evals), symbol{i}, 'MarkerSize', markersize, 'LineWidth', linewidth);
    hold on;
end
hold off;
set(gca, 'FontSize', fontsize, 'LineWidth', linewidth);
legend(names, 'LineWidth', linewidth);
title('Scree Plot');
xlabel('');
ylabel('|Eigenvalue|');
xlim([0 neigs]);
if print_figs
    print_figure(gcf, width, height, [graph_name '-eigs']);
end

%% Compute the LCC for each graph
for i = 1:3
    [~, sizes] = components(graphs{i});
    lcc{i} = max(sizes);
end

%% Print Stuff
fprintf('Graph, Nodes, Edges, LCC %%, GCC\n');
for i = 1:3
    nnodes = sum( full(sum(graphs{i},2)) > 0 );
    fprintf('%s, %d, %d, ', names{i}, nnodes, nnz(graphs{i}));
    fprintf('%2d, ', round( 100 * lcc{i} / nnodes ));
    fprintf('%.2f', gcc{i});
    fprintf('\n');
end

function print_figure(figure_handle, width, height, fname)
set(gcf,'InvertHardcopy','on');
set(gcf, 'PaperUnits', 'inches');
papersize = get(figure_handle, 'PaperSize');
left = (papersize(1)- width)/2;
bottom = (papersize(2)- height)/2;
myfiguresize = [left, bottom, width, height];
set(figure_handle, 'PaperPosition', myfiguresize);
print('-dpng','-r300', fname);

function [ccpd,gcc] = clustering_measures(G)
if ~exist('clustering_coefficients.m','file')
    error('Must install MATLAB_BGL toolbox');
end
options.undirected = 1;
options.unweighted = 1;
cc = clustering_coefficients(G,options);
d = full(sum(G,2));
tf = (d>0);
ccpd = accumarray(d(tf), cc(tf), [max(d) 1], @mean);
gcc = sum( cc .* d .* (d-1) ) / sum( d .* (d-1) );
