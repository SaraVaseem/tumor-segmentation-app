%Your File Path
fig_path = '';
disp(['Folder path: ', fig_path]);

% Verify the folder exists
if isfolder(fig_path)
    disp('Folder exists.');
else
    disp('Folder does not exist. Check the path.');
    return;  % Stop the script
end



% List .mat files
fig_files = dir(fullfile(fig_path, '*.mat'));
if isempty(fig_files)
    disp('No .mat files found. Checking for hidden files...');
    fig_files = dir(fullfile(fig_path, '*.*'));  % List all files
    fig_files = fig_files(~[fig_files.isdir]);  % Exclude folders
    disp('All files (including hidden):');
    disp({fig_files.name}');
end

% Extract numeric part of filenames and sort
file_names = {fig_files.name};  % Get filenames as a cell array
file_numbers = cellfun(@(x) str2double(regexp(x, '\d+', 'match', 'once')), file_names);  % Extract numbers
[~, idx] = sort(file_numbers);  % Sort by numeric value
fig_files = fig_files(idx);  % Reorder files

% Check if .mat files are found
if isempty(fig_files)
    disp('No .mat files found in the specified folder.');
else
    disp(['Found ', num2str(length(fig_files)), ' files.']);
end

% Get the current working directory (where the script is executing from)
output_folder = pwd;
disp(['Current working directory: ', output_folder]);

% Ensure the output folder exists
if ~exist(output_folder, 'dir')
    mkdir(output_folder);  % Create the folder if it doesn't exist
end

name = 'Tumor';

for i = 1:length(fig_files)
    baseFileName = fig_files(i).name;
    fullFileName = fullfile(fig_files(i).folder, baseFileName);
    disp(['Processing file: ', fullFileName]);

    % Load the .mat file
    try
        load(fullFileName);
        disp('File loaded successfully.');
    catch ME
        disp(['Failed to load file: ', fullFileName]);
        disp(['Error: ', ME.message]);
        continue;  % Skip to the next file
    end

    % Verify the structure of the loaded .mat file
    if ~exist('cjdata', 'var') || ~isfield(cjdata, 'image')
        disp(['Skipping file: ', baseFileName, ' (invalid structure)']);
        continue;  % Skip to the next file
    end

    

    img = cjdata.image;  % Assuming the image is stored in cjdata.image
    disp(['Image size: ', num2str(size(img))]);

    % Create filenames for the images
    s = string(i);
    fname = strcat(name, '',s, '.png');

    % Process the image
    try
        im = rescale(img);
    catch ME
        disp(['Error processing image in file: ', baseFileName]);
        disp(['Error: ', ME.message]);
        continue;  % Skip to the next file
    end

    % Save the images to the current working directory
    try
        imwrite(im, fullfile(output_folder, fname));  % Save the original image
        disp(['Successfully saved: ', fullfile(output_folder, fname)]);
    catch ME
        disp(['Failed to save: ', fullfile(output_folder, fname)]);
        disp(['Error: ', ME.message]);
    end
end