clc ;
clear all ;
close all ;

addpath('npy-matlab');
savepath;   % 可选
save_debug_npy = true;
debug_dir = 'debug';
if save_debug_npy && ~exist(debug_dir, 'dir')
    mkdir(debug_dir);
end

%img_uint8 = imread('./images/paper_200_v1_test_1590.png');
%img_uint8 = imread('paper_200_v2_test_723.png');
%img_uint8 = imread('rock_200_v1_test_1484.png');
img_uint8 = imread('scissors_200_v1_test_1644.png');
if ndims(img_uint8) == 3
    img_uint8 = rgb2gray(img_uint8);
end
if ~isequal(size(img_uint8,1), 64) || ~isequal(size(img_uint8,2), 64)
    img_uint8 = imresize(img_uint8, [64, 64]);
end
imshow(img_uint8, []);
%S = load('v2_ori.int8.params.mat');
S = load('v2.int8.params.mat');
disp(S(1).names)
size(S(1).values)

in_zp = int32(S.input_zero_points{1}(1));
if isfield(S, 'activation_zero_points')
    z_conv1_out = int32(S.activation_zero_points{1}(1));
    z_conv2_out = int32(S.activation_zero_points{3}(1));
    z_conv3_out = int32(S.activation_zero_points{5}(1));
else
    z_conv1_out = int32(-128);
    z_conv2_out = int32(-128);
    z_conv3_out = int32(-128);
end

% x_q = round(x/s + zp); with x=uint8/255 and s~=1/255, this is effectively uint8 + zp.
img_q_i32 = int32(img_uint8) + in_zp;
img_q_i32 = max(min(img_q_i32, int32(127)), int32(-128));
img_int8 = int8(img_q_i32);
imshow(img_int8, []);
if save_debug_npy
    writeNPY(img_int8, fullfile(debug_dir, 'matlab_input_q.npy'));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% W1 = S.values{9};                 % 预计 size: [4 3 3 1]
% W1_hwio = permute(W1, [2 3 4 1]); % -> [3 3 1 4]
% 
% k = 1;
% K = W1_hwio(:,:,1,k);             % 3x3
% imagesc(K); axis image; colorbar; title("conv1 kernel out " + k);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%Kernel load%%%%%%%%%%%%%%%%%%%%%%%%%
%----------first conv---------
W1 = S.values{9};
W1_hwio = permute(W1, [2 3 4 1]);   % [3 3 1 4]
b1 = S.values{8};          % 形状通常是 1×4 或 4×1
b1 = int32(b1(:));         % 变成 4×1 的 int32（每个 oc 一个 bias）
%draw 
% figure; tiledlayout(1,4);
% for k = 1:4
%     nexttile;
%     imagesc(W1_hwio(:,:,1,k)); axis image off; clim([-128 127]); colorbar;
%     title("out " + k);
% end
%-----------------------------

%----------second conv---------
W2 = S.values{7};   % 先填正确 index
W2_hwio = permute(W2, [2 3 4 1]);   % 变成 [3 3 4 8]
b2 = S.values{6};   % 对应 bias
b2 = int32(b2(:));
%draw
% figure('Units','normalized','Position',[0.05 0.05 0.9 0.85]);
% t = tiledlayout(8,4,'TileSpacing','compact','Padding','compact');
% for oc = 1:8
%     for ic = 1:4
%         ax = nexttile;
% 
%         K = double(W2_hwio(:,:,ic,oc));
%         Kbig = imresize(K, 40, 'nearest');   % 放大40倍
% 
%         imagesc(ax, Kbig, [-128 127]);
%         axis(ax,'image');
%         axis(ax,'off');
% 
%         title(ax, sprintf("oc %d, ic %d", oc, ic), 'FontSize', 10);
%     end
% end
% colormap(t.Parent, 'parula');
% colorbar;  
%-----------------------------------------

%----------third conv---------
W3 = S.values{5};   % 先填正确 index
W3_hwio = permute(W3, [2 3 4 1]);   % 变成 [3 3 4 8]
b3 = S.values{4};   % 对应 bias
b3 = int32(b3(:));
% %draw
% figure('Units','normalized','Position',[0.05 0.05 0.9 0.85]);
% t = tiledlayout(8,8,'TileSpacing','compact','Padding','compact');
% for oc = 1:8
%     for ic = 1:8
%         ax = nexttile;
% 
%         K = double(W3_hwio(:,:,ic,oc));
%         Kbig = imresize(K, 40, 'nearest');   % 放大40倍
% 
%         imagesc(ax, Kbig, [-128 127]);
%         axis(ax,'image');
%         axis(ax,'off');
% 
%         title(ax, sprintf("oc %d, ic %d", oc, ic), 'FontSize', 10);
%     end
% end
% colormap(t.Parent, 'parula');
% colorbar;  
%-----------------------------------------
%%%%%%%%%%%%%%%%%%%%%%kernel load finished%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%loading scales for weigh%%%%%%%%%%%%%%%%%%
sw1 = S.scales{9};
% Load scales for the second convolution layer
sw2 = S.scales{7};
sw3 = S.scales{5};
sw4 = S.scales{3};

s_in = S.input_scales{1};
s_out=S.output_scales{1};

s_conv1_out = S.activation_scales{1};
s_conv2_out = S.activation_scales{3};
s_conv3_out = S.activation_scales{5};
s_fc_out = S.activation_scales{8};

[M1, qm1, shift1] = tflite_quantize_multiplier(s_in, sw1, s_conv1_out);
[M2, qm2, shift2] = tflite_quantize_multiplier(s_conv1_out, sw2, s_conv2_out);
[M3, qm3, shift3] = tflite_quantize_multiplier(s_conv2_out, sw3, s_conv3_out);
[M4, qm4, shift4] = tflite_quantize_multiplier(s_conv3_out, sw4, s_fc_out);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%loading finisded

%%%%%%%%%%%%%%%%%%%%%%%%%%%starting computaion%%%%%%%%%%%%%%%%%%%
%----------------------------CONV1------------------------------
conv1_img = conv2D_int8(img_int8, W1_hwio, b1,1,'valid',in_zp,int32(0));

figure;
tiledlayout(1,4);
for k = 1:4
    nexttile;
    imshow(conv1_img(:,:,k), []);  % [] 自动拉伸显示
    title("ch " + k);
end
%--------------------requant-------------
conv1_img_int8 = requant_int32_to_int8(conv1_img, qm1, shift1, z_conv1_out);
figure;
tiledlayout(1,4);
for k = 1:4
    nexttile;
    imshow(conv1_img_int8(:,:,k), []);  % [] 自动拉伸显示
    title("ch " + k);
end
% %---------------Relu1-------------
Relu1=relu(conv1_img_int8, int8(z_conv1_out));
if save_debug_npy
    writeNPY(Relu1, fullfile(debug_dir, 'matlab_conv1_relu.npy'));
end
figure;
tiledlayout(1,4);
for k = 1:4
    nexttile;
    imshow(Relu1(:,:,k), []);  % [] 自动拉伸显示
    title("ch " + k);
end
%-----------------pooling1---------
Pooling1=relu_maxpool2x2_int8(Relu1, int8(z_conv1_out));
figure;
tiledlayout(1,4);
for k = 1:4
    nexttile;
    imshow(Pooling1(:,:,k), []);  % [] 自动拉伸显示
    title("ch " + k);
end
%--------------------CONV2-----------------------------
conv2_img = conv2D_int8(Pooling1, W2_hwio, b2,1,'valid',z_conv1_out,int32(0));

figure;
tiledlayout(2,4);

for k = 1:8
    nexttile;
    imshow(conv2_img(:,:,k), []);   % [] 自动拉伸显示
    title("conv2 ch " + k);
end
%--------------------requant-------------
conv2_img_int8 = requant_int32_to_int8(conv2_img, qm2, shift2, z_conv2_out);
figure;
tiledlayout(2,4);
for k = 1:8
    nexttile;
    imshow(conv2_img_int8(:,:,k), []);  % [] 自动拉伸显示
    title("ch " + k);
end
% %---------------Relu2-------------
Relu2=relu(conv2_img_int8, int8(z_conv2_out));
if save_debug_npy
    writeNPY(Relu2, fullfile(debug_dir, 'matlab_conv2_relu.npy'));
end
figure;
tiledlayout(2,4);
for k = 1:8
    nexttile;
    imshow(Relu2(:,:,k), []);  % [] 自动拉伸显示
    title("ch " + k);
end
%-----------------pooling2---------
Pooling2=relu_maxpool2x2_int8(Relu2, int8(z_conv2_out));
figure;
tiledlayout(2,4);
for k = 1:8
    nexttile;
    imshow(Pooling2(:,:,k), []);  % [] 自动拉伸显示
    title("ch " + k);
end
%------------------------------------

%--------------------CONV3-----------------------------
conv3_img = conv2D_int8(Pooling2, W3_hwio, b3,1,'valid',z_conv2_out,int32(0));

figure;
tiledlayout(2,4);

for k = 1:8
    nexttile;
    imshow(conv3_img(:,:,k), []);   % [] 自动拉伸显示
    title("conv3 ch " + k);
end
%--------------------requant-------------
conv3_img_int8 = requant_int32_to_int8(conv3_img, qm3, shift3, z_conv3_out);
figure;
tiledlayout(2,4);
for k = 1:8
    nexttile;
    imshow(conv3_img_int8(:,:,k), []);  % [] 自动拉伸显示
    title("ch " + k);
end
% %---------------Relu3-------------
Relu3=relu(conv3_img_int8, int8(z_conv3_out));
if save_debug_npy
    writeNPY(Relu3, fullfile(debug_dir, 'matlab_conv3_relu.npy'));
end
figure;
tiledlayout(2,4);
for k = 1:8
    nexttile;
    imshow(Relu3(:,:,k), []);  % [] 自动拉伸显示
    title("ch " + k);
end
%-----------------pooling3---------
Pooling3=relu_maxpool2x2_int8(Relu3, int8(z_conv3_out));
figure;
tiledlayout(2,4);
for k = 1:8
    nexttile;
    imshow(Pooling3(:,:,k), []);  % [] 自动拉伸显示
    title("ch " + k);
end
%------------------------------------


%----------------Flatten--------------------
x_fc = flatten_nhwc_int8(Pooling3);     % 1x288 int8 (NHWC顺序)
w_fc = S.values{3};
bfc = S.values{2};
x_zp_fc_in = S.activation_zero_points{7};
w_zp_fc = int32(0);
z_out_fc = S.activation_zero_points{8};
% Wfc_oi: int8 [3,288]
% bfc:    int32 [3,1]
% x_zp_fc, w_zp_fc, qm_fc, shift_fc, z_out_fc

[out_fc_i32, out_fc_i8] = fully_connected_int8( ...
     x_fc, w_fc, bfc, x_zp_fc_in, w_zp_fc, qm4, shift4, z_out_fc);
if save_debug_npy
    writeNPY(int8(x_fc), fullfile(debug_dir, 'matlab_flatten_i8.npy'));
    writeNPY(int8(out_fc_i8), fullfile(debug_dir, 'matlab_dense_i8.npy'));
    writeNPY(int32(out_fc_i32), fullfile(debug_dir, 'matlab_dense_i32.npy'));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% out_fc_i8: 3x1 或 1x3 的 int8 logits
[~, idx] = max(double(out_fc_i8));   % idx 是 1-based

class_names = ["paper", "rock", "scissors"];
pred_label = class_names(idx);

fprintf('pred idx (MATLAB 1-based): %d\n', idx);
fprintf('pred label: %s\n', pred_label);
