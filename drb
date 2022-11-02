# drb方法的主要实现代码，分为两类：batch_size=1时的单个音频文件推理阶段；batch_size>1时的微调训练阶段

def amax_poolProb(alist,prob,batchsize,maxlen):
    alist = alist.tolist()
    result = []
    alist.append(-1)
    alength = len(alist)-1
    s_s,e_s = 0,0
    while s_s < alength:
        if alist[s_s] + 1 == alist[s_s+1]:
            i = s_s
            while (i < alength) and (alist[i] + 1 == alist[i+1]) and  (alist[i+1] % maxlen != 0):
                i += 1
            e_s = i
            tempp, tempd = prob[alist[s_s]:alist[e_s] + 1].topk(1)
            border_f = alist[s_s] + tempd.item()
            s_s = e_s + 1
        else:
            border_f = alist[s_s]
            s_s += 1
        result.append(border_f)
    return result

# batch_size>1
def encoder_drb(encoder_out: torch.Tensor):
    # 数据形状
    # import time
    # s_time = time.time()
    batchsize = encoder_out.size(0)
    maxlen = encoder_out.size(1)
    encoder_dim = encoder_out.size(2)
    device = encoder_out.device
    # 提取非blank帧
    # 1、获取概率
    ctc_pro = self.ctc.log_softmax(encoder_out)
    topk, topk_index = ctc_pro.topk(1, dim=2)
    # 2、根据blank分段
    topk_index = topk_index.squeeze(-1)  # (B,maxlen)
    topk_index = topk_index.view(-1)
    topk = topk.squeeze(-1)
    topk = topk.view(-1)
    arange = torch.arange(0, batchsize * maxlen, dtype=torch.int64, device=device)
    # 2.1 非blank部分
    utemp = topk_index != 0
    utemp = arange[utemp]
    # 2.2 blank部分
    ntemp = topk_index == 0
    ntemp = arange[ntemp]
    # 2.3 取边界帧,连续空格取概率最大的帧
    rb = amax_poolProb(ntemp, topk, batchsize, maxlen)
    cindex = torch.cat((utemp, torch.tensor(rb, dtype=utemp.dtype, device=device)))
    cindex, _ = torch.sort(cindex)  # 排序使得边界帧被正确插入
    i = 1
    startt = 0
    lengtha = []
    for value in cindex:
        if value >= i * maxlen:
            lengtha.append(startt)
            startt = 1
            i += 1
        else:
            startt += 1
    lengtha.append(startt)
    newmax = max(lengtha)
    # 新的length (B,)
    new_alength = torch.tensor(lengtha, dtype=torch.int64, device=device)
    # 提取到的关键帧编码器信息
    encoder_out = torch.index_select(encoder_out.view(-1, encoder_dim), dim=0, index=cindex)
    # 填充提取的数据
    newencoder_out = torch.zeros((batchsize, newmax, encoder_dim),
                                dtype=encoder_out.dtype, device=device)
    encoder_mask = ~make_pad_mask(new_alength, newmax).unsqueeze(1)
    start = 0
    for i, j in enumerate(new_alength):
        j = j.item()
        newencoder_out[i, :j] = encoder_out[start:start + j, ]
        start += j
    return newencoder_out, encoder_mask

# batch_size=1
def encoder_d(encoder_out: torch.Tensor):
    # batchsize = 1
    batchsize = encoder_out.size(0)
    maxlen = encoder_out.size(1)
    device = encoder_out.device
    assert batchsize == 1
    # 提取非blank帧
    # 1、获取概率
    ctc_pro = self.ctc.log_softmax(encoder_out)
    topk, topk_index = ctc_pro.topk(1, dim=2)
    # 2、根据blank分段
    topk_index = topk_index.view(-1)  # (maxlen,)
    topk = topk.view(-1)
    arange = torch.arange(0, maxlen, dtype=torch.int64, device=device)
    # 2.1 非blank部分
    utemp = topk_index != 0  # (b,t)
    utemp = arange[utemp]
    # 2.2 blank部分
    ntemp = topk_index == 0
    ntemp = arange[ntemp]
    # 2.3 取边界帧,连续空格取概率最大的帧
    ntemp = ntemp.tolist()
    ntemp.append(-1)
    start_sb, end_sb = 0, 0
    border_result = []
    templength = len(ntemp) - 1
    while start_sb < templength:
        if ntemp[start_sb + 1] == ntemp[start_sb] + 1:
            i = start_sb
            while (i < templength) and (ntemp[i + 1] == ntemp[i] + 1):
                i += 1
            end_sb = i
            tempp, tempd = topk[ntemp[start_sb]:ntemp[end_sb] + 1].topk(1)
            border_f = ntemp[start_sb] + tempd.item()
            border_result.append(border_f)
            start_sb = end_sb + 1
        else:
            border_result.append(ntemp[start_sb])
            start_sb += 1
    # 合并取到的边界帧
    cindex = torch.cat((utemp, torch.tensor(border_result, dtype=utemp.dtype, device=device)))
    cindex, _ = torch.sort(cindex)  # 排序使得边界帧被正确插入
    # 提取到的关键帧编码器信息
    encoder_out = torch.index_select(encoder_out.squeeze(0), dim=0, index=cindex)
    encoder_out = encoder_out.unsqueeze(0)
    new_alength = torch.tensor(cindex.size(0), dtype=torch.int64,device=device)
    new_alength = new_alength.unsqueeze(0)
    encoder_mask = ~make_pad_mask(new_alength).unsqueeze(1)
    # e_time = time.time()-s_time
    return encoder_out, encoder_mask
