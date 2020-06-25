from ctdr.utils import util_vis, util_mesh, render_objs
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
# from mesh_intersection.loss import DistanceFieldPenetrationLoss

from ctdr.utils import util_vis, util_mesh

def get_params(model, exclude_mus=False):
    return model.parameters()

def keep_topology(model, search_tree, displace_prev):
    vertices = model.get_displaced_vertices()
    faces = model.faces
    triangles = vertices[faces].unsqueeze(dim=0)
    
    with torch.no_grad():
        torch.cuda.synchronize()
        collision_idxs = search_tree(triangles)
        torch.cuda.synchronize()
        
        output = collision_idxs.squeeze()
        collisions = output[output[:, 0] >= 0, :]
        #print("@ total number of collisions:", collisions.shape[0])
        mask_coll_faces = model.labels[collisions[:, 0],:] != model.labels[collisions[:, 1],:]
        mask_coll_faces = torch.prod(mask_coll_faces, dim=1)==1
        collisions = collisions[mask_coll_faces, :]
        
    if collisions.shape[0] == 0:
        return None, None
    else:
        print("@ collisions: ", collisions.shape[0])
    
    all_intr_verts = []
    all_recv_verts = []
    
    # case 2: topology-preserving constraint
    cnt = 0
    while collisions.shape[0] > 0 and cnt <= 400:
        cnt += 1
        assert cnt <= 300
        with torch.no_grad():
            intr_verts_vx3_int = model.faces[collisions[:,0]] # face number
            recv_verts_vx3_int = model.faces[collisions[:,1]]

            intr_verts = torch.unique(intr_verts_vx3_int)
            recv_verts = torch.unique(recv_verts_vx3_int)
            
            all_intr_verts.append(intr_verts)
            all_recv_verts.append(recv_verts)
            
            #idx_verts = torch.cat([intr_verts-1, intr_verts+1, intr_verts, recv_verts-1, recv_verts+1, recv_verts], dim=0)
            idx_verts = torch.cat([intr_verts, recv_verts], dim=0)
            idx_verts = torch.clamp(idx_verts, 0, model.vertices.shape[0]-1)
            
            #model.displace.data[idx_verts,:] = (model.displace.data[idx_verts,:]+displace_prev[idx_verts,:]) / 2.0
            model.displace.data[idx_verts,:] = displace_prev[idx_verts,:]
            # roll-back
            #model.displace.data = model.displace.data - args.lr * model.displace.grad.data
            
            # double check
            vertices = model.vertices + model.displace
            #vertices = model.get_displaced_vertices()
            triangles = vertices[faces].unsqueeze(dim=0)
        
            torch.cuda.synchronize()
            collision_idxs = search_tree(triangles)
            torch.cuda.synchronize()

            output = collision_idxs.squeeze()
            collisions = output[output[:, 0] >= 0, :]
            #print("@ collisions including self-intersec: ", collisions.shape[0])
            #print(triangles.shape, collision_idxs.shape)

            mask_coll_faces = model.labels[collisions[:, 0],:] != model.labels[collisions[:, 1],:]
            #print(mask_coll_faces)
            mask_coll_faces = torch.prod(mask_coll_faces, dim=1)==1
            #print(mask_coll_faces)
            collisions = collisions[mask_coll_faces, :]
            
            print("@ collisions: ", collisions.shape[0])
            nc = collisions.shape[0]
            #assert(nc == 0)
            #collisions = collision_idxs
    
    all_intr_verts = torch.unique(torch.cat(all_intr_verts, dim=0))
    all_recv_verts = torch.unique(torch.cat(all_recv_verts, dim=0))
    
    vertices = model.get_displaced_vertices()
    
    return vertices[all_intr_verts,:], vertices[all_recv_verts,:]
    

def run(model, ds, niter, args, epoch_start=0, use_adam=True):
    use_repulsion = False

    use_collision = True
    if args.data[-1]=="A" or ds.nmaterials == 2:
        use_collision = False

    if use_collision:
        from mesh_intersection.bvh_search_tree import BVH
        search_tree = BVH(max_collisions=16)
        #pen_distance = DistanceFieldPenetrationLoss(sigma=0.5)

    print("@ model.mus", model.mus)
    
    params = get_params(model)
    
    if use_adam == True:
        opt = torch.optim.Adam(params, args.lr, betas=(0.9, 0.99))
    else:
        opt = torch.optim.SGD(params, lr=args.lr, momentum=0.9, nesterov=True)

#     loop = tqdm.tqdm(list(range(0,args.niter)))

    if epoch_start == 0:
        f = open(args.dresult+'log.txt', 'w')
    else:
        f = open(args.dresult+'log.txt', 'a')
    
    log = f"@ statistics of mesh: {model.vertices.shape[0]}, {model.faces.shape[0]}\n"
    
    # full-batch case
    if args.b == 0:
        idx_angles_full = torch.LongTensor(np.arange(ds.nangles))
        p_full = ds.p.cuda()
        ds_loader = [ [ idx_angles_full, p_full ]  ]

    # mini-batch case
    else:
        ds_loader = torch.utils.data.DataLoader(ds, batch_size=args.b, shuffle=True,
        num_workers=0, drop_last=True, pin_memory=True)

    mask_bg = ds.p < 1e-5
    mask_bg = mask_bg.cuda()
    use_silhouette = False
    # if use_silhouette:
    #     mask_bg = ds.p < 1e-5
    #     mask_bg = mask_bg.cuda()

#     mask_bg = (p_batch > p_full.min()+0.05)
    #mask_bg = 1
    ledge = 0
    llap = 0.
    lflat = 0.

    for epoch in range(epoch_start, niter):
        if epoch+100 == niter and niter > 400:
           args.lr *= 0.5
           print("@ args.lr", args.lr)
        
        # if epoch % 20 == 0 or epoch == niter-1:
        start = time.time()
        
        for idx_angles, p_batch in ds_loader:
            displace_prev = model.displace.data.clone()
            if args.b > 0:
                p_batch = p_batch.cuda()

            opt.zero_grad()
            
            phat, mask_valid, edge_loss, lap_loss, flat_loss = model(idx_angles, args.wedge) # full angles
            # phat[~mask_valid] = 0.0
            # mask_valid = mask_valid + mask_bg
            
            if 1:
                # l2 loss
                data_loss = (p_batch - phat)[mask_valid].pow(2).mean()

                if use_silhouette:
                    idx_bg = (~mask_valid) * mask_bg[idx_angles]
                    nbg = torch.sum(idx_bg)
                    if nbg:
                        print("sum(idx_bg), min, max", nbg, torch.min(phat[idx_bg]).item(), torch.max(phat[idx_bg]).item())
                        # print(phat[idx_bg])
                    
                        data_loss += (phat[idx_bg]).pow(2).mean()
                        # data_loss += 10 * torch.abs(phat[idx_bg]).mean()

                # add for the invalid pixels
                # data_loss += (p_batch)[~mask_valid].pow(2).mean()
            else:
                # student t misfit
                sigma=1
                data_loss = torch.log(1 + ((p_batch - phat)[mask_valid]/sigma)**2).mean()

            loss = data_loss + args.wedge * edge_loss + args.wlap * lap_loss + args.wflat * flat_loss
            
            
            if use_collision and use_repulsion:
                wrep = 1e-3
                loss += wrep * loss_repulsion

#             v1, v2 = model.get_points()
#             pt_dist = kaolin.metrics.point.directed_distance(v1, v2, True)
#             loss += (pt_dist-0.1)**2
            
            loss.backward()
            opt.step()
            
            loss_now = loss.item()
            model.mus.data.clamp_(min=0.0)
        
            if use_collision == False:
                continue
            use_repulsion = False
            

            # intr, recv = keep_topology(model, search_tree, displace_prev)
            # # intr, recv: vertices indices
            # # if there are some intruders, we add repulsion term
            # if intr is not None:
            #     assert(0)
            #     import kaolin
            #     d12 = kaolin.metrics.point.directed_distance(intr, recv, False)
            #     d21 = kaolin.metrics.point.directed_distance(recv, intr, False)
                
            #     d12 = d12*d12 
            #     d21 = d21*d21
            #     #kaolin.metrics.point.chamfer_distance(intr, recv)
            #     #vertices = model.get_displaced_vertices()
            #     #dist = torch.sum((vertices[intr,:]-vertices[recv,:])**2, dim=1)
            #     #dist = torch.sqrt(dist)
            #     #assert(0)
            #     loss_repulsion = torch.mean(1. / (d12 + 1e-10))
            #     loss_repulsion += torch.mean(1. / (d21 + 1e-10))
            #     loss_repulsion *= 0.0001
            #     print(f"loss_repulsion: {loss_repulsion}")	
            #     # use_repulsion = True
                
#             v1, v2 = model.get_points()
#             pt_dist = kaolin.metrics.point.directed_distance(v1, v2, False)
#             print(f"min directed distance is {pt_dist.min()} shape: {pt_dist.shape}")
            
        
#         if epoch == niter // 2:
#             for param_group in opt.param_groups:
#                 param_group['lr'] = args.lr * 0.5

        # if epoch % 20 == 0 or epoch == niter-1:            
        elpased_time = time.time() - start

        if epoch > epoch_start and args.b == 0:
            dloss = (loss_prev - loss_now) # should be positive
            if dloss < 1e-11 and dloss > 0:
                print('! converged')
                break
            
        loss_prev = loss_now
        

        if args.wedge > 0.:
            ledge = edge_loss.item()
            
        if args.wlap > 0.:
            llap = lap_loss.item()
        
        if args.wflat > 0.:
            lflat = flat_loss.item()

        log += f'~ {epoch:03d} l2_loss: {data_loss.item():.8f} edge: {ledge:.6f} lap: {llap:.6f} flat: {lflat:.6f} mus: {str(model.mus.cpu().detach().numpy())} time: {elpased_time:.4f}\n'
        #log += f'center: {model.center[0,0].item():.6f} {model.center[0,1].item():.6f} {model.center[0,2].item():.6f}'
        # f.write(log+"\n")
            
        if epoch % 60 == 0 or epoch == niter-1:
            if torch.sum(~mask_valid) > 15000 and epoch > 100:
                assert 0, "consider increasing regularization"

            print(log)

            if args.b == 0:
                res_np = ds.p_original - phat.detach().cpu().numpy()                
                res_scalar = np.mean(res_np**2)
                f.write(f"~ res_np: {res_scalar}\n")
                util_vis.save_sino_as_img(args.dresult+f'{epoch:04d}_sino_res.png', res_np, cmap='coolwarm')
            
            phat[~mask_valid]=0.
            print(phat.min(), phat.max())
            if args.verbose == 0:
                continue
				
            vv = model.vertices.cpu()+model.displace.detach().cpu()
            ff = model.faces.cpu()
            
            labels_v, labels_f = model.labels_v_np, model.labels.cpu().numpy()
            # util_vis.save_vf_as_img_labels(args.dresult+f'{epoch:04d}_render.png', vv, ff, labels_v, labels_f)
            util_vis.save_sino_as_img(args.dresult+f'{epoch:04d}_sino.png', phat.detach().cpu().numpy())
            util_mesh.save_mesh(args.dresult+f'{epoch:04d}.obj', vv.numpy(), ff.numpy(), labels_v, labels_f)
            if args.data == "3nanoC":
                import subprocess
                subprocess.Popen(["python", "../plot/compute_volume.py", args.dresult+f'{epoch:04d}.obj'])

            if epoch == niter-1:
                util_mesh.save_mesh(args.dresult+'mesh.obj', vv.numpy(), ff.numpy(), labels_v, labels_f)
                util_vis.save_sino_as_img(args.dresult+f'{epoch:04d}_data.png', ds.p.cuda())
        
    f.write(log+"\n")

def update_topology(model):
    # update topology information
    mask_coll_faces = model.labels[collisions[:, 0]] != model.labels[collisions[:, 1]]
    mask_coll_faces = torch.sum(mask_coll_faces, dim=1)
    fidx_list_intruder = collisions[mask_coll_faces, 0]
    fidx_list_receiver = collisions[mask_coll_faces, 1]
    #assert len(mask_coll_faces)==1, 

    ncolls2 = mask_coll_faces.sum().item()
    if ncolls2 == 0:
        return

    print("number of collisions2:", ncolls2)

    # swap the outside labels for intruder and receiver
    labels_coll1 = model.labels[fidx_list_intruder,1]
    model.labels[fidx_list_intruder,1] = model.labels[fidx_list_receiver,1]
    model.labels[fidx_list_receiver,1] = labels_coll1
