% This file is to loop through the json files of the labelme annotations to remove the mask overlay from the encoded image 
name = "Tumor";

% Set to range of images you want to extract the data from
for i = 1400:1532

    %Needed to skip specific files that caused errors in our data set
    if(i~=30 && i~=43 && i~=73 && i~=117 && i~=186)
        % Read the JSON file into a struct
        s = string(i); % Change to i for all files
        fname = strcat(name, '', s, '.png'); % Image file name
        fname2 = strcat(name, '', s, '.json'); % Output JSON file name
        fname3 = strcat(name, '', s, '.txt'); % Output text file name (if needed)
    
        object = readstruct(fname2);


    % Call the Python function and pass the image path
        encodedImageData = pyrunfile("ImgEncode.py", "encoded_image", path=fname);
    
    % Convert the Python string to a MATLAB string
        encodedImageData = string(encodedImageData); % or use char(encodedImageData)
    
    % Assign the encoded image data to the object
        object.imageData = encodedImageData;
    
    % Ensure the object matches the LabelMe format
        labelmeStruct = struct();
        labelmeStruct.version = object.version;
        labelmeStruct.flags = object.flags;
        labelmeStruct.shapes = {object.shapes}; % Wrap the shape in a cell array
        labelmeStruct.imagePath = object.imagePath; % Image filename
        labelmeStruct.imageData = object.imageData; % Base64-encoded image data
        labelmeStruct.imageHeight = object.imageHeight; % Image height
        labelmeStruct.imageWidth = object.imageWidth;

    % Convert the struct to a JSON string
        toSave = jsonencode(labelmeStruct, "PrettyPrint", true);
    
    % Save the JSON string to a file
        fid = fopen(fname2, 'w');
        fprintf(fid, '%s', toSave);
        fclose(fid);
    end
end