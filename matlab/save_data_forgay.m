% ===== conv1_img : int8, size could be HxW or HxWxC =====
x = conv1_img;

% 展平成 1D，建议用 column-major（MATLAB默认） or 你自己定义 row-major
% 方案A：直接 MATLAB 默认顺序（最简单，但要确保 TB 用同样的索引映射）
v = x(:);

fid = fopen('conv1_img.mem', 'w');
assert(fid>0);

for i = 1:numel(v)
    u = uint8(v(i));              % 保留二补码位型
    fprintf(fid, '%02X\n', u);    % 8-bit hex
end
fclose(fid);