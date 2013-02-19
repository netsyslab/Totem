#! /usr/bin/octave -qf

% Creates a scale-free network using bter random graph generator.
%
% The graph is saved as an octave spare matrix, which has the following format:
%  # name: G
%  # type: sparse matrix
%  # nnz: <number of edges>
%  # rows: <number of vertices>
%  # columns: <number of vertices>
%  <source node id> <destination node id> <connected>
%
% Example:
% ./scale-free.m 1 3 /tmp/test.mat; cat /tmp/test.mat
% # Created by Octave 3.4.3
% # name: G
% # type: sparse matrix
% # nnz: 8
% # rows: 6
% # columns: 6
% 3 1 1
% 4 2 1
% 1 3 1
% 2 4 1
% 5 4 1
% 4 5 1
% 6 5 1
% 5 6 1
%
% Created on: 2013-02-16
% Author: Abdullah Gharaibeh

% Get command line arguments
arg_list = argv();
if length(arg_list) != 3
  fprintf("\nError: missing arguments\n");
  fprintf("usage: %s <alpha> <maxdeg> <output_file>\n", program_name());
  exit(-1)
end
alpha = str2double(arg_list{1});
maxdeg = str2num(arg_list{2});
saveat = arg_list{3};

% Create a power-law distribution of edge degree
ddist = round(maxdeg^(alpha)./(1:maxdeg).^(alpha))';

% Invoke BTER generator and save the generated sparse matrix
G = bter(ddist);
save(saveat, "G");
