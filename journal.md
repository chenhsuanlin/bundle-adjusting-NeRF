
## BARF developments

Starting date : 7th April, 2022


19th April, 2022

* Planar Image Alignment - I just started training


12th April, 2022

* Check rotation and translation / compare them with GT
* Can you do novel view synthesis? yes - but we need to understand the pose more - to do better
    * How does NeRF do novel view synthesis? How does it fix the pose?


* with full PE, BARF still works OK... 


11th April, 2022

* Manually render all training images - it seems to work well
* How can you deal with novel view synthesis? (how to obtain the novel view from this pose)
* Is there any possibility that the model just fit to the training scene? 



10th April, 2022

* Train the Freiburg Cars again - with proper focal length, compare the result with NeRF
* After the training, I am going to train the model with MAE
* I could not train Vanilla NeRF - modify the learning rate 
    * Randomly sample 1024 rays - and do 200K iterations : 13 iterations for 1 full iter - so it is similar to 15K full iters
    * Fix the learning rate as 5e-4 and train for 200K iters

Things 2 try

* Tomorrow, I should try manually render all training images
* To understand how BARF align coordinates for unseen views
* MAE + BARF try

#### Code Issues

* model: train.py
    * base.py: train_epoch
    * base.py: train_iteration
    * base.py: graph.forward, graph.compute_loss, graph.summarize_loss
        * nerf.py (Graph-forward) : self.get_pose, self.render(opt.render.rand_rays : then randomly sample)
        * nerf.py (con't) : self.render - camera.get_center_ray(grid->homogeneous->camera-coord->world-coord)
        * nerf.py (con't) : self.sample_depth - sample points along the ray
        * nerf.py (nerf) : self.nerf.forward_samples, self.nerf.composite
        * nerf.py (Graph-compute_loss) / nerf.py(Graph - summarize_loss)
        
        * barf.py (self.get_pose): var.se3_refine = self.se3_refine.weight[var.idx] # find the pose and put there
        * barf.py  pose_refine = camera.lie.se3_to_SE3(var.se3_refine) -> From se(3), recover R and T - http://jinyongjeong.github.io/Download/SE3/jlblanco2010geometry3d_techrep.pdf
        * barf.py  in pose_refine, Rotation seems to be axis-angle representation (mapped in the R^3), Translation needs to check
        * barf.py pose = camera.pose.compose([pose_refine,pose]) -> update the initial pose(for Freiburg, it is torch.eye(3))
        * sim3 matrix -> https://robotics.stackexchange.com/questions/19305/in-slam-loop-closure-what-does-a-sim3-matrix-do
* evaluate.py : resume is required
        

* add learned pose correction to all training data - it may not be updating one by one
* graph : render, camera: given pose and intrinsic, return the rays, for these rays, render using graph
* can you put GPU as an argument?
* saved_dir : output/GROUP/NAME
* network : m.graph 
* tqdm - log - there are many hidden gems here
* visdom port check - it is possible to access from server
* focal was wrong - it was modified
* Use 'iphone' for Freiburg cars - no - stick to LLFF/Blender

#### History

* with full PE

python train.py --group=freiburg_cars --model=barf --yaml=barf_freiburg_cars --name=car001_barf_full_pe --data.scene=car001

* batch_size=2048, constant lr rate, try again

python evaluate.py --group=freiburg_cars --model=barf --yaml=barf_freiburg_cars --name=car001_barf_pe_lr_test --data.scene=car001 --resume

python train.py --group=freiburg_cars --model=barf --yaml=barf_freiburg_cars --name=car001_barf_pe_lr_test --data.scene=car001 --barf_c2f="[0.1, 0.5]"
python train.py --group=freiburg_cars --model=nerf --yaml=nerf_freiburg_cars --name=car001_nerf_lr_test --data.scene=car001 

python evaluate_train.py --group=freiburg_cars --model=barf --yaml=barf_freiburg_cars --name=car001_barf_pe_lr_test --data.scene=car001 --resume
python evaluate_train.py  --group=freiburg_cars --model=nerf --yaml=nerf_freiburg_cars --name=car001_nerf_lr_test --resume

python train.py --group=freiburg_cars --model=barf --yaml=barf_freiburg_cars --name=car001_barf_pe_lr_test --data.scene=car001 --barf_c2f="[0.1, 0.5]"
python train.py --group=freiburg_cars --model=nerf --yaml=nerf_freiburg_cars --name=car001_nerf_lr_test --data.scene=car001 

python train.py --group=freiburg_cars --model=barf --yaml=barf_freiburg_cars --name=car001_barf_pe --data.scene=car001 --barf_c2f="[0.1, 0.5]"
python train.py --group=freiburg_cars --model=nerf --yaml=nerf_freiburg_cars --name=car001_nerf --data.scene=car001 




### Check the pre-trained models

Some questions

1) How does BARF initialize the pose?

* All zeros

2) How does BARF train the model so quickly? (especially compared to my implementation)

* Randomly sample 1024 pixels and update the model

3) What does the validation set mean here? - I think there may not be a validation set for BARF

* I need to dig-in the code - the validation here means that just checking 4 first or last images 

4) Does BARF have coarse and fine network?

* No

5) How is the PE at BARF different from others?

* The training progress is normalized between 0 and 1 - 
* when it is above start, then the high frequency part could be applied through weight
* keep the progress part so we can easily apply BARF style PE





#### Running Freiburg cars dataset

As far as I know, the dataset has the same structure as LLFF. Both rely on COLMAP to get the posed images and sparse depth points.

llff : llff / (fern/flower/fortress/horns/leaves/orchids/room/trex) / (images, images_4, images_8, outputs, sparse, colmap_output.txt, database.db,poses_bounds.npy)

freiburg_cars / (car001) / (images, images_6, sparse, colmap_output.txt, poses_bounds.npy, train_test_splits.json, database.db)


#### Running Dataset

opt - options.py
add new files on data



