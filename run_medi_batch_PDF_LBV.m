% run_medi_batch_PDF_LBV.m
% =========================================================================
% Background field removal by PDF and LBV, from the MEDI toolbox.
%
% Processes every in_*.mat in the exchange folder and writes one out_*.mat
% per dataset holding both results. The Python side reads these names:
%
%   RDF_PDF    local field after PDF, in radians (divide by 2*pi for Hz)
%   Mask_PDF   region in which PDF produced valid output
%   RDF_LBV    local field after LBV, same units
%   Mask_LBV   LBV's valid region
%
% qsm_pipeline.py uses RDF_PDF and Mask_PDF only. The LBV outputs are
% consumed by twin_method_comparison.py, which compares V-SHARP, PDF and LBV
% against the digital twin's ground truth.
%
% BEFORE RUNNING
% --------------
% 1. Set MEDI_PATH to your own copy of the MEDI toolbox, available from
%    Cornell's MRI Research Lab under their licence.
% 2. Set EXCHANGE to the folder the Python side wrote in_*.mat into.
% =========================================================================

% ---- MEDI toolbox location ----
MEDI_PATH = 'C:\path\to\MEDI_toolbox';

% ---- Exchange folder: whatever the Python side was configured to write
% ---- into. <RESULTS_ROOT>/MEDI_EXCHANGE for the measured pipeline, or the
% ---- twin comparison's own exchange folder for the twin run.
EXCHANGE = 'C:\path\to\MEDI_EXCHANGE';

% =========================================================================

addpath(genpath(MEDI_PATH));

if ~exist('PDF', 'file') || ~exist('LBV', 'file')
    error(['PDF or LBV not found on the MATLAB path. Check that MEDI_PATH ' ...
           'points at the MEDI toolbox root: %s'], MEDI_PATH);
end

if ~isfolder(EXCHANGE)
    error('Exchange folder does not exist: %s', EXCHANGE);
end

files = dir(fullfile(EXCHANGE, 'in_*.mat'));
if isempty(files)
    error(['No in_*.mat found in %s. Run the Python side first ' ...
           '(MODE = "prep").'], EXCHANGE);
end
fprintf('found %d datasets in %s\n\n', numel(files), EXCHANGE);

n_done = 0;
n_failed = 0;

for f = 1:numel(files)
  try
    inpath = fullfile(EXCHANGE, files(f).name);
    tag = files(f).name(4:end-4);          % strip 'in_' and '.mat'
    outpath = fullfile(EXCHANGE, ['out_' tag '.mat']);

    fprintf('=== %s ===\n', tag);
    S = load(inpath);

    iFreq = double(S.iFreq);
    Mask = double(S.Mask);
    N_std = double(S.N_std);
    matrix_size = double(S.matrix_size(:)');
    voxel_size = double(S.voxel_size(:)');
    B0_dir = double(S.B0_dir(:)');

    fprintf(' %dx%dx%d, voxel [%.3f %.3f %.3f] mm, mask %d voxels\n', ...
        matrix_size(1), matrix_size(2), matrix_size(3), ...
        voxel_size(1), voxel_size(2), voxel_size(3), nnz(Mask));
    fprintf(' field map %.1f to %.1f Hz\n', ...
        min(iFreq(Mask>0))/(2*pi), max(iFreq(Mask>0))/(2*pi));

    % ---------------------------------------------------------------------
    % PDF
    % ---------------------------------------------------------------------
    t0 = tic;
    RDF_PDF = PDF(iFreq, N_std, Mask, matrix_size, voxel_size, B0_dir);
    Mask_PDF = Mask;                       % PDF keeps the whole mask
    tPDF = toc(t0);
    lf = RDF_PDF(Mask_PDF>0)/(2*pi);
    fprintf(' PDF ran in %.1f s\n', tPDF);
    fprintf(' PDF  local field %.2f to %.2f Hz, sd %.3f Hz, cov %.1f%%\n', ...
        min(lf), max(lf), std(lf), 100*nnz(Mask_PDF)/nnz(Mask));

    % Convergence check. PDF should remove most of the field range. If the
    % output range is comparable with the input range, the solver has not
    % converged and the local field is unusable, whatever its coverage.
    in_range  = max(iFreq(Mask>0))   - min(iFreq(Mask>0));
    out_range = max(RDF_PDF(Mask>0)) - min(RDF_PDF(Mask>0));
    if out_range > 0.5*in_range
        fprintf([' WARNING: PDF output range is %.0f%% of the input range.\n' ...
                 '          The solver may not have converged. Check its\n' ...
                 '          iteration count above before using this result.\n'], ...
                 100*out_range/in_range);
    end

    % ---------------------------------------------------------------------
    % LBV
    % ---------------------------------------------------------------------
    % LBV solves a boundary value problem inside the mask. Its arguments
    % differ from PDF's: tolerance, depth of the multigrid, and peel, which
    % is how many voxels it strips from the boundary. peel = 0 keeps the
    % full mask, which is the fair comparison against PDF; the MEDI default
    % of 1 or 2 would hand PDF an advantage in coverage that is a setting
    % rather than a property of the method.
    t0 = tic;
    tol = 0.01;
    depth = -1;          % -1 lets LBV choose the multigrid depth
    peel = 0;            % keep the whole mask, see above
    RDF_LBV = LBV(iFreq, Mask, matrix_size, voxel_size, tol, depth, peel);
    tLBV = toc(t0);

    % LBV returns zeros where it could not solve; treat those as invalid
    Mask_LBV = Mask .* (RDF_LBV ~= 0);
    lfl = RDF_LBV(Mask_LBV>0)/(2*pi);
    fprintf(' LBV ran in %.1f s\n', tLBV);
    fprintf(' LBV  local field %.2f to %.2f Hz, sd %.3f Hz, cov %.1f%%\n', ...
        min(lfl), max(lfl), std(lfl), 100*nnz(Mask_LBV)/nnz(Mask));

    % ---------------------------------------------------------------------
    % -v7, not -v7.3: scipy.io.loadmat cannot read v7.3, which is HDF5.
    % MATLAB can promote a bare save() to v7.3 for large variables, so the
    % version is set explicitly.
    save(outpath, 'RDF_PDF', 'Mask_PDF', 'RDF_LBV', 'Mask_LBV', ...
         'matrix_size', 'voxel_size', 'B0_dir', '-v7');
    fprintf(' wrote out_%s.mat\n\n', tag);
    n_done = n_done + 1;

  catch ME
    % One dataset failing should not abandon the rest of the batch. Only
    % successful datasets get an out_*.mat, so the Python side will report
    % the missing ones.
    fprintf(' FAILED: %s\n\n', ME.message);
    n_failed = n_failed + 1;
  end
end

fprintf('=====================================================\n');
fprintf('done: %d of %d datasets', n_done, numel(files));
if n_failed > 0
    fprintf(', %d FAILED (see above)', n_failed);
end
fprintf('.\n');
fprintf('Return to the Python side: qsm_pipeline.py with\n');
fprintf('MODE = "invert" for the measured data, or\n');
fprintf('twin_method_comparison.py for the twin run.\n');
fprintf('=====================================================\n');