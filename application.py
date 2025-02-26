import argparse
import math
import random
import time
from io import BytesIO
from pathlib import Path
from typing import Tuple
import os
# use relative paths
os.chdir(os.path.dirname(__file__))

import cv2
import matplotlib
import nerfview
import numpy as np
import torch
import viser
from gsplat.rendering import rasterization
from nerfstudio.cameras.cameras import Cameras as nerfCameras
from nerfstudio.utils.eval_utils import eval_setup
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from scipy.spatial.transform import Rotation
from torch import optim
from torchvision.ops.boxes import masks_to_boxes
from torchvision.transforms.v2 import functional as Fv2
from tqdm import tqdm

from gui.sam2_helper import *
from gui.sam2_helper import self_prompt
from gui.style_helper import *
from gui.util import *

parser = argparse.ArgumentParser()
parser.add_argument(
    "--output_dir", type=str, default="results/", help="where to dump outputs"
)
parser.add_argument(
    "--scene_grid", type=int, default=1, help="repeat the scene into a grid of NxN"
)
parser.add_argument("--ckpt", type=str, default="", help="path to the .pt file")
parser.add_argument("--port", type=int, default=8080, help="port for the viewer server")
parser.add_argument(
    "--backend", type=str, default="gsplat", help="gsplat, gsplat_legacy, inria"
)

parser.add_argument(
    "--config",
    type=str,
    default="outputs/garden_processed/splatfacto/2025-01-11_223202/config.yml",
    
)

args = parser.parse_args()
assert args.scene_grid % 2 == 1, "scene_grid must be odd"

config = args.config

torch.manual_seed(42)
device = "cuda"

if args.ckpt != "":
    ckpt = torch.load(args.ckpt, map_location=device)
    # all require grad
    means = ckpt["means"].requires_grad_(True)
    quats = ckpt["quats"].requires_grad_(True)
    scales = torch.exp(ckpt["scales"].requires_grad_(True))
    opacities = torch.sigmoid(ckpt["opacities"].requires_grad_(True)).squeeze(-1)
    features_dc = ckpt["features_dc"].requires_grad_(True)
    colors = torch.sigmoid(features_dc).squeeze(-1)
    sh_degree = None


def viewer():
    port = args.port
    ViserViewer(port)


class ViserViewer:
    def __init__(self, port):
        """
        self.ckpt = torch.load(args.ckpt, map_location=device)
        self.means = self.ckpt["means"]
        self.quats = self.ckpt["quats"]
        self.scales = torch.exp(self.ckpt["scales"])
        self.opacities = torch.sigmoid(self.ckpt["opacities"]).squeeze(-1)
        self.features_dc = self.ckpt["features_dc"]
        self.colors = torch.sigmoid(self.features_dc).squeeze(-1)
        self.og_colors = self.colors.clone()
        self.sh_degree = None
        """

        self.show_mask_points = False
        self.save_render_img = False
        self.display_style = False
        self.latest_camera_state = None  # raus machen
        self.final_mask = None
        self.display_style_progress = False
        self.show_mask = False
        self.pcl_size = None

        # Multi object
        self.object_id_runner = 0
        self.objects = {}
        self.choose_points_object_id = -1
        self.mask_points_2d = []
        self.object_mask_img = []
        self.object_points_pcl = []

        # Config file of trained nerfstudio model
        load_config = Path(config)

        self.config, self.pipeline, _, self.step = eval_setup(
            load_config,
            eval_num_rays_per_chunk=None,
            test_mode="test",
        )

        self.cameras = self.pipeline.datamanager.train_dataset.cameras
        self.image_dataset = self.pipeline.datamanager.train_dataset
        self.model = self.pipeline.model

        self.model.eval()  # Needed?


        if args.ckpt != "":
            self.means = means
            self.quats = quats
            self.scales = scales
            self.opacities = opacities
            self.features_dc = features_dc
            self.colors = colors
            self.sh_degree = None
        else:
            self.means = self.model.gauss_params["means"]
            self.quats = self.model.gauss_params["quats"]
            self.scales = torch.exp(self.model.gauss_params["scales"])
            self.opacities = torch.sigmoid(self.model.gauss_params["opacities"]).squeeze(-1)
            self.features_dc = self.model.gauss_params["features_dc"]
            self.colors = torch.sigmoid(self.features_dc).squeeze(-1)
            self.sh_degree = None

    
        print("Viewer is running")
        self.port = port
        self.server = viser.ViserServer(host="localhost", port=port)
        self.server.on_client_connect(self.initialize_camera)

        self.render = self.viewer_render_fn
        # self.server.request_share_url()
        self.viewer = nerfview.Viewer(
            server=self.server,
            render_fn=self.render,
            mode="rendering",
        )

        self.xyz = self.model.means

        self.mask_points = []
        self.mask_img = None

        checkpoint = "./libs/sam2/checkpoints/sam2.1_hiera_tiny.pt"
        model_cfg = "/configs/sam2.1/sam2.1_hiera_t.yaml"

        self.predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

        # UI Style
        self.server.gui.configure_theme(
            # control_width="small",
            show_logo=False,
            control_layout="collapsible",
            dark_mode=True,
            brand_color=(187, 134, 252),
        )

        self.server.gui.set_panel_label("Object-Wise Style Transfer")

        with self.server.gui.add_folder("Preprocessing Images...Please wait.", order=0) as self.processing_folder:
            self.preprocess_progress = self.server.gui.add_progress_bar(
                0, visible=False, order=1, color="green"
            )

        # self.tab_group = self.server.gui.add_tab_group()
        # self.view_tab = self.tab_group.add_tab("View", viser.Icon.CAMERA)
        # self.objects_tab = self.tab_group.add_tab("Objects", viser.Icon.CUBE)
        #
        # with self.view_tab:
        with self.server.gui.add_folder("View") as self.view_folder:
            self.viewer._rendering_folder.remove()
            self.viewer._max_img_res_slider = self.server.gui.add_slider(
                "Max Img Res", min=64, max=2048, step=1, initial_value=2048,
                hint="A lower resolution decreases rendering time but may reduce image quality."
            )
            self.viewer._max_img_res_slider.on_update(self.viewer.rerender)

            # self.preprocess_btn = self.server.gui.add_button("Preprocess")
            # self.preprocess_progress = self.server.gui.add_progress_bar(0, visible=False)
            self.camera_slider = self.server.gui.add_slider(
                "Camera Viewpoint",
                min=0,
                max=len(self.cameras) - 1,
                step=1,
                initial_value=0,
                hint="View the scene from the training Camera viewpoints",
            )
            @self.camera_slider.on_update
            def _(guiHandle):
                camera_idx = self.camera_slider.value
                # print(f"Current camera index: {camera_idx}")
                clientHandle = guiHandle.client
                self.update_camera(camera_idx, clientHandle)

            # self.show_camera_frustums = self.server.gui.add_button(
            #     "Show Cameras", icon=viser.Icon.EYE
            # )

            export_btn = self.server.gui.add_button("Export Scene...", icon=viser.Icon.DOWNLOAD)
            export_btn.on_click(self.export)

        # if True:#with self.objects_tab:
        with self.server.gui.add_folder("Objects") as self.objects_folder:
            self.add_object_btn = self.server.gui.add_button(
                "Add Object", icon=viser.Icon.PLUS
            )
            self.obj_list_tabs = self.server.gui.add_tab_group()
            # self.server.gui.add_button(
            #     "Info-Button",
            #     hint="Dies ist ein hilfreicher Hinweistext.",
            #     icon=viser.Icon.INFO_CIRCLE
            # )

        # self.preprocess_btn.on_click(self.preprocess_click)
        self.add_object_btn.on_click(self.add_object)

        # self.show_camera_frustums.on_click(self.toggle_camera_frustums)

        self.add_object(None)

        print("Viewer running... Ctrl+C to exit.")

        self.preprocess_click(None)

        time.sleep(100000)

    def initialize_camera(self, clientHandle):
        print("Initializing camera")
        self.update_camera(0, clientHandle)

    def update_camera(self, camera_idx, clientHandle):
        camera = self.cameras[camera_idx]

        c2w = camera.camera_to_worlds

        # Extract position
        position = c2w[:3, 3]
        # Extract rotation
        rotation = c2w[:3, :3]
        up_direction = rotation[:, 1]
        look_dir = -rotation[:, 2]

        look_at = first_axis_intersection(position, look_dir)
        # look_at = position + look_dir * 0.05

        rotation = Rotation.from_matrix(rotation)
        wxyz = rotation.as_quat()
        clientHandle.camera.wxyz = np.array([wxyz[3], wxyz[0], wxyz[1], wxyz[2]])

        # Set camera orientation
        clientHandle.camera.position = position
        clientHandle.camera.look_at = look_at  # np.array([0.0, -0.00, -0.01])
        clientHandle.camera.up_direction = up_direction  # np.array([0.0, 1.0, 1.0])

        # Set FOV from camera intrinsics (vertical FOV in radians, see _viser.py)
        fy = camera.fy
        height = camera.image_height
        fov = 2 * math.atan(height / (2 * fy))
        clientHandle.camera.fov = fov
        clientHandle.camera.near = 0.00001
        clientHandle.camera.far = 1000.0

    def toggle_camera_frustums(self, guiHandle):
        if not hasattr(self, "camera_frustums_visible"):
            self.camera_frustums_visible = False
        self.camera_frustums_visible = not self.camera_frustums_visible
        on = self.camera_frustums_visible
        if on and not hasattr(self, "camera_frustums"):
            self.add_camera_frustums()
        elif on and hasattr(self, "camera_frustums"):
            for frustum in self.camera_frustums:
                frustum.visible = True
        elif not on and hasattr(self, "camera_frustums"):
            for frustum in self.camera_frustums:
                frustum.visible = False

        # guiHandle.target.label = "Hide Cameras" if on else "Show Cameras"
        # guiHandle.target.icon = viser.Icon.EYE if on else viser.Icon.EYE_OFF

    def add_camera_frustums(self):
        """
        name: str,
        fov: float,
        aspect: float,
        scale: float = 0.3,
        line_width: float = 2.0,
        color: RgbTupleOrArray = (20, 20, 20),
        image: np.ndarray | None = None,
        format: Literal["png", "jpeg"] = "jpeg",
        jpeg_quality: int | None = None,
        wxyz: tuple[float, float, float, float] | np.ndarray = (1.0, 0.0, 0.0, 0.0),
        position: tuple[float, float, float] | np.ndarray = (0.0, 0.0, 0.0),
        visible: bool = True,"""

        self.camera_frustums = []
        for i, camera in enumerate(self.cameras):
            c2w = camera.camera_to_worlds.numpy()

            position = c2w[0, :3, 3]
            rotation = c2w[0, :3, :3]
            rotation = Rotation.from_matrix(rotation)
            wxyz = rotation.as_quat()

            fx = camera.fx[0].item()

            width = camera.width[0].item()
            height = camera.height[0].item()

            aspect = width / height
            fov = 2 * math.atan(width / (2 * fx))

            scale = 0.1
            line_width = 2.0
            color = (20, 20, 20)
            image = self.image_dataset.get_image_float32(i).detach().cpu().numpy()
            format = "jpeg"
            jpeg_quality = None
            visible = True

            self.camera_frustums.append(
                self.server.scene.add_camera_frustum(
                    f"Cameras/{i:05d}",
                    fov,
                    aspect,
                    scale,
                    line_width,
                    color,
                    image,
                    format,
                    jpeg_quality,
                    wxyz,
                    position,
                    visible,
                )
            )

    def add_object(self, _):
        # Get Object ID
        object_id = self.object_id_runner
        self.object_id_runner += 1

        indicator_color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        )

        self.objects[object_id] = {"indicator color": indicator_color}

        self.mask_points_2d.append([])
        self.object_mask_img.append(None)
        self.object_points_pcl.append(None)


        with self.obj_list_tabs.add_tab(f"Obj {object_id}") as object_tab:
            
          with self.server.gui.add_folder(f"Segmentation") as segmentation_folder:
            choose_points_btn = self.server.gui.add_button(
                "Select Points", hint="Click on the object you want to mask"
            )
            done_choosing_btn = self.server.gui.add_button(
                "Done Selecting",
                visible=False,
                hint="Your points will be projected to 3D",
            )

            if hasattr(self, "preprocessing_done") and self.preprocessing_done:
                hint_segmentation = "Select points to segment the object"
            else:
                hint_segmentation = (
                    "Wait for preprocessing to finish to segment the object"
                )

            segment_gaussians_btn = self.server.gui.add_button(
                "Start Segmentation", disabled=True, hint=hint_segmentation
            )
            segment_gaussians_progress = self.server.gui.add_progress_bar(
                0, visible=False
            )
            ensemble_threshold = self.server.gui.add_slider(
                "Segmentation Threshold",
                min=0,
                max=1,
                step=0.05,
                initial_value=0.5,
                visible=False,
                hint="Change this to fine-tune the segmentation",
            )
            highlight_mask_btn = self.server.gui.add_checkbox(
                "Show Mask", True, visible=False
            )
            self.server.gui.add_markdown(
                content="<details><summary>ℹ <small>Hints for Better Segmentation</small></summary><small><b>Select points</b> that clearly define the object boundaries. Adjust the <b>segmentation threshold</b> to refine the segmentation mask. A lower threshold might include more of the background, while a higher threshold can focus on specific features. <b>Pro Tip:</b> For complex objects, select multiple points to improve segmentation quality.</small></details>"
                )            

          with self.server.gui.add_folder(f"Style Transfer") as style_folder:

            placeholder_img = np.zeros((2, 2, 3), dtype=np.uint8)
            style_img_display = self.server.gui.add_image(
                placeholder_img, label="Style image", format="jpeg", visible=False
            )

            upload_style_btn = self.server.gui.add_upload_button(
                "Upload Style Image", icon=viser.Icon.UPLOAD, mime_type="image/*"
            )           
            match_colors_btn = self.server.gui.add_button(
                "Match Colors",
                disabled=True,
                visible=False,
            )
            style_strength_slider = self.server.gui.add_slider(
                "Style Strength",
                min=0.0,
                max=2.0,
                step=0.05,
                initial_value=1.0,  # default 100% Style
                hint="Change this to adjust the strength of the style transfer",
                visible=False,
            )

            color_vs_style_slider = self.server.gui.add_slider(
                "Color vs Style balance",
                min=0.0,
                max=1.0,
                step=0.05,
                initial_value=0.5,
                hint="Change this to adjust the balance between color and style matching, 0.0 for more color matching, 1.0 for more style matching",
            )

            iterations_input = self.server.gui.add_text(
                "Style Iterations", "2000",
                hint="Number of iterations for the style transfer. A higher number may improve the results."
            )

            transfer_style_btn = self.server.gui.add_button(
                "Transfer Style",
                disabled=True,
                hint="Transfer the style of the uploaded image to the selected object",
            )
            transfer_style_progress = self.server.gui.add_progress_bar(0, visible=False)
            
            self.server.gui.add_markdown(
                content="<details><summary>ℹ <small>Hints for Better Style Transfer</small></summary><small><b>Color vs Style:</b> Adjust this slider to balance color matching (0.0) and stylization (1.0). <b>Pro Tip:</b> If the output looks too abstract, try reducing the stylization and gradually increasing the iterations to refine the details.</small></details>"
                )

        
          with self.server.gui.add_folder(f"Control") as segmentation_folder:
            delete_object_btn = self.server.gui.add_button(
                f"Delete Object {object_id}", icon=viser.Icon.TRASH
            )

            
        choose_points_btn.on_click(
            lambda guiHandle: self.choose_points(
                guiHandle, object_id, done_choosing_btn
            )
        )
        done_choosing_btn.on_click(
            lambda guiHandle: self.done_choosing(
                guiHandle, object_id, choose_points_btn, segment_gaussians_btn
            )
        )
        segment_gaussians_btn.on_click(
            lambda guiHandle: self.segment_gaussians(
                guiHandle,
                object_id,
                segment_gaussians_progress,
                highlight_mask_btn,
                ensemble_threshold,
                transfer_style_btn,
            )
        )
        upload_style_btn.on_upload(
            lambda _: self.upload_style(
                upload_style_btn, object_id, style_img_display, transfer_style_btn, match_colors_btn
            )
        )

        transfer_style_btn.on_click(
            lambda guiHandle: self.transfer_style(
                guiHandle, object_id, transfer_style_progress, highlight_mask_btn, style_strength_slider, iterations_input, color_vs_style_slider
            )
        )
        ensemble_threshold.on_update(
            lambda guiHandle: self.segmentation_threshold_update(guiHandle, object_id)
        )
        delete_object_btn.on_click(lambda _: self.delete_object(object_id, object_tab))
        match_colors_btn.on_click(lambda _: self.match_colors(object_id, highlight_mask_btn))

        self.objects[object_id]["segment_gaussians_btn"] = segment_gaussians_btn

        @highlight_mask_btn.on_update
        def _(_):
            self.objects[object_id]["highlight mask"] = highlight_mask_btn.value
            self.viewer.rerender("hello world")

        # done_choosing_btn.on_click(self.done_choosing(object_id))
        # segment_gaussians_btn.on_click(self.segment_gaussians(object_id))
        # transfer_style_btn.on_click(self.transfer_style(object_id))

    # ----------------------------------------------------------------------------------------------------------
    # -------------------------------------------- MULTI OBJECT NEW SHITTT -------------------------------------
    # ----------------------------------------------------------------------------------------------------------

    def on_click(self, event):
        print(f"Mouse click at {event.screen_pos}")
        if self.choose_points_object_id == -1:
            return

        object_id = self.choose_points_object_id

        points = self.objects[object_id].get("mask 2d points", [])
        points.append([event.screen_pos[0][0], event.screen_pos[0][1]])

        self.objects[object_id]["mask 2d points"] = points

        # self.mask_points_2d[self.choose_points_object_id].append([event.screen_pos[0][0], event.screen_pos[0][1]])

        # self.mask_points.append([event.screen_pos[0][0], event.screen_pos[0][1]]) # TODO delete later

        self.viewer.rerender("hello world")

    def choose_points(self, guiHandle, object_id, done_choosing_btn):
        # delete old points and pcl if they exist
        # if self.mask_points_2d[object_id]:
        #    self.mask_points_2d[object_id] = []
        #    self.server.scene.remove_point_cloud(self.object_points_pcl[object_id])

        self.camera_slider.disabled = True
        self.camera_slider.hint = "The camera slider is disabled while selecting points."
        print("Select points")
        self.show_mask_points = True
        self.choose_points_object_id = object_id
        # cant move scene anymore aaaahhhhhhhh
        self.original_callback = self.server.scene._scene_pointer_cb
        self.server.scene.on_pointer_event("click")(self.on_click)

        done_choosing_btn.visible = True
        guiHandle.target.visible = False

        return

    def done_choosing(
        self, guiHandle, object_id, choose_points_btn, segment_gaussians_btn
    ):
        client = guiHandle.client

        # project 2d points to 3d

        # restore original callback
        self.server.scene.remove_pointer_callback()
        self.show_mask_points = False
        self.viewer.rerender("hello world")

        client_camera = client.camera

        mask_cam = self.client_cam_2_nerfstudio_cam(client_camera)
        # self.objects[object_id]["mask_cam"] = mask_cam

        # self.mask_cam = self.client_cam_2_nerfstudio_cam(client_camera)

        nerfview_cam = self.client_cam_2_nerfview_cam(client_camera)

        # get aspect
        aspect = client_camera.aspect
        width = 1000
        height = int(width / aspect)

        # render camera image
        render_image = self.viewer_render_fn(nerfview_cam, (width, height))
        print("Render image shape: ", render_image.shape)

        # save mask image
        # self.object_mask_img[object_id] = (render_image.copy() * 255).astype(np.uint8)
        # self.objects[object_id]["mask_img"] = (render_image.copy() * 255).astype(np.uint8)

        # show prompts in 3d
        nerfstudio_cam = mask_cam

        width = nerfstudio_cam.width[0].item()
        height = nerfstudio_cam.height[0].item()
        input_points = []

        for point in self.objects[object_id]["mask 2d points"]:
            input_points.append([int(point[0] * width), int(point[1] * height)])

        if len(input_points) == 0:
            print("No points chosen")
            # enable choosing again
            guiHandle.target.visible = False
            choose_points_btn.visible = True
            self.choose_points_object_id = -1
            return

        # for point in self.mask_points:
        #    input_points.append([int(point[0] * width), int(point[1] * height)])

        # convert 2d points to 3d prompts
        prompts_3d = generate_3d_prompts(self.xyz, nerfstudio_cam, input_points)

        og_prompts_3d = self.objects[object_id].get("3d prompts", None)
        if og_prompts_3d is not None:
            prompts_3d = torch.cat((prompts_3d, og_prompts_3d), dim=0)

        prompts_3d_np = prompts_3d.detach().cpu().numpy()

        self.objects[object_id]["3d prompts"] = prompts_3d

        if self.pcl_size is None:
            self.pcl_size = abs(
                self.compute_point_size(prompts_3d_np, nerfstudio_cam, 3)
                .detach()
                .cpu()
                .numpy()[0][0]
            )
            print("PCL size: ", self.pcl_size)

        self.objects[object_id]["prompt pcl"] = self.server.scene.add_point_cloud(
            f"3D Prompts Object {object_id}",
            points=prompts_3d_np,
            colors=self.objects[object_id].get("indicator color", (255, 0, 0)),
            point_size=self.pcl_size,
            point_shape="circle",
        )
        print("3D prompts added to scene")

        # enable choosing again
        guiHandle.target.visible = False
        choose_points_btn.visible = True

        self.camera_slider.disabled = False
        self.camera_slider.hint = "View the scene from the training Camera viewpoints."

        if hasattr(self, "preprocessing_done") and self.preprocessing_done:
            segment_gaussians_btn.disabled = False
            segment_gaussians_btn.hint = "Segment 3D object from the scene"

        self.objects[object_id]["mask 2d points"] = []
        self.choose_points_object_id = -1        
        return

    def segment_gaussians(
        self,
        guiHandle,
        object_id,
        progress_bar,
        highlight_mask_btn,
        ensemble_threshold,
        transfer_style_btn,
    ):
        print("Start GS Segmentation...")

        # generate 3D prompts
        prompts_3d = self.objects[object_id]["3d prompts"]
        print("3D prompts: ", prompts_3d)

        multiview_masks = []
        sam_masks = []

        obj_cameras = []

        progress_bar.visible = True
        max_progress = len(self.cameras)
        for i, view in tqdm(enumerate(self.cameras)):
            progress_bar.value = (i / max_progress) * 100
            prompts_2d = project_to_2d(self.cameras[i : i + 1], prompts_3d)

            valid_prompts = []
            for j, prompt in enumerate(prompts_2d):
                if prompt[0] >= 0 and prompt[1] >= 0:
                    if (
                        prompt[0] < view.image_width.item()
                        and prompt[1] < view.image_height.item()
                    ):
                        valid_prompts.append(j)
            prompts_2d = prompts_2d[valid_prompts]

            if not valid_prompts:  # Object is not visible in this view
                continue

            obj_cameras.append(i)
            # sam prediction
            sam_mask = self_prompt(self.predictor, prompts_2d, self.sam_features[i])
            if len(sam_mask.shape) != 2:
                sam_mask = torch.from_numpy(sam_mask).squeeze(-1)  # .to("cuda")
            else:
                sam_mask = torch.from_numpy(sam_mask)  # .to("cuda")
            sam_mask = sam_mask.long()
            sam_masks.append(sam_mask)

            # mask assignment to gaussians
            point_mask, indices_mask = mask_inverse(
                self.xyz, self.cameras[i : i + 1], sam_mask
            )

            multiview_masks.append(point_mask.unsqueeze(-1))
            # # gaussian decomposition as an intermediate process
            # if args.gd_interval != -1 \
            #                     and i % args.gd_interval == 0:  #
            #     gaussians = gaussian_decomp(gaussians, view, sam_mask, indices_mask)

        # multi-view label ensemble
        progress_bar.animated = True

        self.objects[object_id]["obj_cameras"] = obj_cameras

        self.objects[object_id]["multiview_masks"] = multiview_masks
        _, final_mask = ensemble(multiview_masks, ensemble_threshold.value)

        self.objects[object_id]["final mask"] = final_mask

        progress_bar.animated = False
        progress_bar.visible = False

        '''
        gd_interval = 1


        class dotdict(dict):
            """dot.notation access to dictionary attributes"""

            __getattr__ = dict.get
            __setattr__ = dict.__setitem__
            __delattr__ = dict.__delitem__


        means = self.model.means.cuda()
        quats = self.model.quats.cuda()
        scales = self.model.scales.cuda()
        colors = self.model.colors.cuda()
        opacities = self.model.opacities.cuda()

        gaussians = {
            "means": means,
            "quats": quats,
            "scales": scales,
            "colors": colors,
            "opacities": opacities,
        }
        gaussians = dotdict(gaussians)

        for i, view in tqdm(enumerate(self.cameras)):
            if gd_interval != -1 and i % gd_interval == 0:
                try:
                    input_mask = sam_masks[i]
                except IndexError:
                    continue
                gaussians = gaussian_decomp(
                    gaussians, self.cameras[i : i + 1], input_mask, final_mask.to("cuda")
                )

        self.means[final_mask] = gaussians.means.detach().cpu()
        self.scales[final_mask] = gaussians.scales.detach().cpu()
        '''
        print("Segmentation Done!")

        print(f"Number of Gaussians in the object mask: {len(final_mask)}")

        self.objects[object_id]["highlight mask"] = True
        highlight_mask_btn.visible = True
        ensemble_threshold.visible = True

        self.objects[object_id]["prompt pcl"].remove()

        if self.objects[object_id].get("style_img", None) is not None:
            transfer_style_btn.disabled = False
        

        if self.objects[object_id].get("matched colors", None) is not None:
            self.objects[object_id].pop("matched colors")
        
        #self.camera_slider.disabled = False


        self.viewer.rerender("hello world")

    def segmentation_threshold_update(self, guiHandle, object_id):
        threshold = guiHandle.target.value

        _, final_mask = ensemble(self.objects[object_id]["multiview_masks"], threshold)
        print(f"Number of Gaussians in the object mask: {len(final_mask)}")

        self.objects[object_id]["final mask"] = final_mask

        if self.objects[object_id].get("matched colors", None) is not None:
            self.objects[object_id].pop("matched colors")

        self.viewer.rerender("hello world")

    def upload_style(
        self, upload_btn, object_id, style_img_display, transfer_style_btn, match_colors_btn
    ):
        print("Upload style image")
        # check if its a valid image
        if not upload_btn.value:
            return
        # check if its a valid image
        style_img = Image.open(BytesIO(upload_btn.value.content))

        # convert to 3 channels
        if style_img.mode != "RGB":
            style_img = style_img.convert("RGB")

        self.objects[object_id]["style_img"] = style_img

        # to numpy
        style_img = np.array(style_img)

        style_img_display.image = style_img
        style_img_display.visible = True

        if self.objects[object_id].get("final mask", None) is not None:
            transfer_style_btn.disabled = False
            match_colors_btn.disabled = False

    def match_colors(self, object_id, highlight_mask_btn): # histogram matching
    
        final_mask = self.objects[object_id].get("final mask", None)
        if final_mask is None:
            print("No final mask available")
            return
        features_dc = (
            self.model.gauss_params["features_dc"]
            .cuda()
            .detach()
            .cpu()
            .numpy()[final_mask]
        )
        # -------------------------------
        #features_dc = self.model.gauss_params["features_dc"].cuda().detach().cpu().numpy()
        ## set final mask to all digits from 0 to features_dc.shape[0]
        #self.objects[object_id]["final mask"] = torch.tensor(
        #    np.arange(features_dc.shape[0])
        #).cuda()
        # ------------------------- only testing
        

        features_dc = 1/(1+(np.exp(-features_dc)))

        #features_dc = matplotlib.colors.rgb_to_hsv(features_dc).astype(np.float32)

        print("Features shape: ", features_dc.shape)

        img = np.array(self.objects[object_id]["style_img"]) / 255
        img = img.reshape(-1, 3)
        #img = matplotlib.colors.rgb_to_hsv(img)
        print("Image shape: ", img.shape)
        print(img)

        print("Image min/max: ", img.min(), img.max())
        print("Features min/max: ", features_dc.min(), features_dc.max())


        for channel in range(3):
            src_hist, src_bin = np.histogram(img[:, channel], bins=256, range=(0, 1))
            dst_hist, dst_bin = np.histogram(features_dc[:, channel], bins=256, range=(0, 1))

            ## save histograms
            #import matplotlib.pyplot as plt
            #plt.plot(src_hist, color="r")
            #plt.plot(dst_hist, color="b")
            #plt.legend(["Image", "Object"])
            #plt.savefig(f"histograms_{channel}.png")
            #plt.close()



            # CDF
            src_cdf = np.cumsum(src_hist)
            src_cdf_normalized = src_cdf / src_cdf.max()

            dst_cdf = np.cumsum(dst_hist)
            dst_cdf_normalized = dst_cdf / dst_cdf.max()

            # save CDFs
            #plt.plot(src_cdf_normalized, color="r")
            #plt.plot(dst_cdf_normalized, color="b")
            #plt.legend(["Image", "Object"])
            #plt.savefig(f"CDFs_{channel}.png")
            #plt.close()

            bin_edges = np.linspace(0, 1, 257)  # 256 bins between [0, 1]
            feature_bins = np.digitize(features_dc[:, channel], bins=bin_edges, right=True) - 1
            feature_bins = np.clip(feature_bins, 0, len(bin_edges) - 2)  # Clamp to valid bin indices

            features_dc[:, channel] = np.interp(
                bin_edges[feature_bins],  # Feature values aligned to bins
                src_cdf_normalized,      # Source CDF
                dst_cdf_normalized       # Target CDF
            )

            #mapping = np.interp(src_cdf_normalized, dst_cdf_normalized, dst_bin[:-1])
            #features_dc[:, channel] = np.interp(features_dc[:, channel], src_bin[:-1], mapping)

            final_hist, _ = np.histogram(features_dc[:, channel], bins=256, range=(0, 1))
            final_cdf = np.cumsum(final_hist)
            final_cdf_normalized = final_cdf / final_cdf.max()
            #plt.plot(src_cdf_normalized, color="r")
            #plt.plot(final_cdf_normalized, color="b")
            #plt.legend(["Image", "Final"])
            #plt.savefig(f"final_cdf_{channel}.png")
            #plt.close()

        
        print("Features: ", features_dc)

        #features_dc = matplotlib.colors.hsv_to_rgb(features_dc).astype(np.float32)

        # inverse sigmoid
        features_dc = np.log(features_dc/(1 - features_dc))

        self.objects[object_id]["matched colors"] = (
            (torch.tensor(features_dc)).cuda().detach()
        )

        print("Matched colors.")

        highlight_mask_btn.value = False
        self.objects[object_id]["highlight mask"] = False
        self.viewer.rerender("hello world")

    def match_colors_hsv(self, object_id, highlight_mask_btn): # hsv mean and std
        final_mask = self.objects[object_id].get("final mask", None)
        features_dc = (
            self.model.gauss_params["features_dc"]
            .cuda()
            .detach()
            .cpu()
            .numpy()[final_mask]
        )

        features_dc = 1/(1+(np.exp(-features_dc)))

        # convert features_dc to HSV
        features_dc_hsv = matplotlib.colors.rgb_to_hsv(features_dc)

        img_hsv = np.array(self.objects[object_id]["style_img"].convert("HSV"))

        img_hsv_flat = img_hsv.reshape(-1, 3)  # Shape: [num_pixels, 3]

        mean_hsv = img_hsv_flat.mean(axis=0)  # Shape: [3]
        std_hsv = img_hsv_flat.std(axis=0)  # Shape: [3]

        mean_hsv = mean_hsv / 255
        std_hsv = std_hsv / 255

        
        # denormalize features_dc_hsv
        features_dc_hsv = features_dc_hsv * (std_hsv) + mean_hsv
        
        # convert back to RGB
        features_dc_rgb = matplotlib.colors.hsv_to_rgb(features_dc_hsv).astype(
            np.float32
        )

        # inverse sigmoid
        features_dc_rgb = np.log(features_dc_rgb/(1 - features_dc_rgb))

        self.objects[object_id]["matched colors"] = (
            (torch.tensor(features_dc_rgb)).cuda().detach()
        )

        print("Matched colors.")

        highlight_mask_btn.value = False
        self.objects[object_id]["highlight mask"] = False
        self.viewer.rerender("hello world")


    def transfer_style(self, guiHandle, object_id, progress_bar, highlight_mask_btn, style_strength_slider, iterations_input, color_vs_style_slider):
        print("Transfer style")


        if self.objects[object_id].get("final mask", None) is None:
            print("No final mask available")
            return
        

        progress_bar.visible = True
        progress_bar.animated = True
        progress_bar.value = 100

        final_mask = self.objects[object_id]["final mask"]

        if self.objects[object_id].get("matched colors", None) is None:
            features_dc_opt = (
                self.model.gauss_params["features_dc"][final_mask.detach()]
                .cuda()
                .detach()
                .clone()
            )
        else:
            print("Using matched colors")
            features_dc_opt = (
                self.objects[object_id]["matched colors"].cuda().detach().clone()
            )
        features_dc_opt.requires_grad_(True)

        means_opt = self.model.means[final_mask].cuda().detach()
        quats_opt = self.model.quats[final_mask].cuda().detach()
        scales_opt = self.model.scales[final_mask].cuda().detach()
        opacities_opt = self.model.opacities[final_mask].cuda().detach()
        # colors.requires_grad_(False)
        optimizer = optim.AdamW([features_dc_opt], lr=0.01)

        H = self.cameras.image_height[0].item()
        W = self.cameras.image_width[0].item()

        # Convert to torch
        style_image = (
            Fv2.pil_to_tensor(self.objects[object_id]["style_img"]).unsqueeze(0).cuda()
        )
        style_image = Fv2.to_dtype(style_image, torch.float32, scale=True)
        style_image = Fv2.resize(style_image, 256)
        nnfm_loss_fn = NNFMLoss("cuda")

        style_strength = style_strength_slider.value
        iters = int(iterations_input.value)
        # style_image = style_image.transpose(2, 3)

        obj_cameras = self.objects[object_id]["obj_cameras"]

        guiHandle.target.disabled = True # disable transfer style button

        style_match = color_vs_style_slider.value

        loss_names = ["nnfm_loss", "gram_loss"]
        loss_max = np.zeros(len(loss_names))

        progress_bar.animated = False
        progress_bar.value = 0
        for i in range(iters):
            progress_bar.value = (i / iters) * 100
            idx = random.choice(obj_cameras)
            viewmats = get_viewmat(self.cameras[idx : idx + 1].camera_to_worlds).cuda()
            Ks = self.cameras[0:1].get_intrinsics_matrices().cuda()
            # gt_img = image_dataset.get_image_float32(idx)
            optimizer.zero_grad(set_to_none=True)

            colors, alpha, meta = rasterization(
                means_opt,  # [N, 3]
                quats_opt,  # [N, 4]
                torch.exp(scales_opt),  # [N, 3]
                torch.sigmoid(opacities_opt).squeeze(-1),  # [N]
                torch.sigmoid(features_dc_opt).squeeze(-1),  # [N, 3]
                viewmats,  # [1, 4, 4]
                Ks,  # [1, 3, 3]
                W,
                H,
                sh_degree=None,
                packed=False,
            )
            colors = colors.permute(0, 3, 1, 2)

            new_obj_mask = (colors > 0).any(dim=1)

            if new_obj_mask.sum() == 0:
                continue

            crop_box = masks_to_boxes(new_obj_mask)[0].to(int)
            new_rgb = colors[:, :, crop_box[1] : crop_box[3], crop_box[0] : crop_box[2]]
            new_rgb = Fv2.resize(new_rgb, style_image.shape[-2:])


            lossdict = nnfm_loss_fn(
                outputs=new_rgb,
                styles=style_image,
                loss_names=loss_names,
            )

            # normalize all losses
            for loss_name, loss in lossdict.items():
                if loss > loss_max[loss_names.index(loss_name)]:
                    loss_max[loss_names.index(loss_name)] = loss.item()
                lossdict[loss_name] = loss / loss_max[loss_names.index(loss_name)]

            loss = style_strength * ((1-style_match)*lossdict["nnfm_loss"] + style_match*lossdict["gram_loss"])
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print(f"iteration {i}, Loss =", loss.item())
                if i % 50 == 0:
                    self.style_progress = (
                        new_rgb[0].cpu().detach().permute(1, 2, 0).numpy() * 255
                    )
                    self.display_style_progress = True
                    self.viewer.rerender("hello world")
                    # plt.imshow(new_rgb[0].cpu().detach().permute(1, 2, 0).numpy())
                    # plt.show()

        # features_dc = self.model.gauss_params["features_dc"].cuda().detach()
        # features_dc[final_mask] = features_dc_opt.detach()

        self.objects[object_id]["styled features dc"] = features_dc_opt.detach()

        # self.colors = torch.sigmoid(features_dc).squeeze(-1).detach()

        self.display_style_progress = False
        progress_bar.visible = False
        self.show_mask = False
        guiHandle.target.disabled = False # disable transfer style button

        self.objects[object_id]["highlight mask"] = False
        highlight_mask_btn.value = False      
        self.viewer.rerender("hello world")

    def delete_object(self, object_id, object_tab):
        print("Delete object")

        pcl = self.objects[object_id].get("prompt pcl", None)
        if pcl != None:
            pcl.remove()

        self.objects.pop(object_id)
        object_tab.remove()

        self.viewer.rerender("hello world")

    # ----------------------------------------------------------------------------------------------------------
    # -------------------------------------------- MULTI OBJECT NEW SHITTT END ---------------------------------
    # ----------------------------------------------------------------------------------------------------------

    def export(self, _):
        print("Exporting...")
        with self.server.gui.add_modal("Export") as modal:
            export_path_btn = self.server.gui.add_text("Export Path", "./checkpoints/styled_scene.pt")
            export_modal_btn = self.server.gui.add_button("Export Scene", icon=viser.Icon.DOWNLOAD)
            cancel_export_btn = self.server.gui.add_button("Cancel", icon=viser.Icon.ARROW_BACK)
            
            @cancel_export_btn.on_click
            def _(_):
                modal.close()

            @export_path_btn.on_update
            def _(_):
                if export_path_btn.value[-3:] != ".pt":
                    time.sleep(1)
                    if export_path_btn.value[-3:] != ".pt":
                        export_path_btn.value = export_path_btn.value + ".pt"

            @export_modal_btn.on_click
            def _(_):
                export_path = export_path_btn.value
                print(f"Exporting to {export_path}")
                
                # only use the folder and make sure it exists
                parent_folder = os.path.dirname(export_path)
                if not os.path.exists(parent_folder):
                    print(f"Creating folder {parent_folder}")
                    os.makedirs(parent_folder)

                features = self.model.gauss_params["features_dc"].cuda().detach()

                for obj in self.objects.values():
                    if obj.get("styled features dc", None) is not None:
                        features[obj["final mask"]] = obj["styled features dc"].cuda().detach()
                print("alive 2")
                save_dict = {
                    "means": self.model.means.cuda().data,
                    "quats": self.model.quats.cuda().data,
                    "scales": self.model.scales.cuda().data,
                    "opacities": self.model.opacities.cuda().data,
                    "features_dc": features.data,
                }
                print(save_dict)

                print(self.model.gauss_params["features_dc"].cuda().detach())

                # check if .pt is at the end of the path
                if export_path[-3:] == ".pt":
                    torch.save(save_dict, export_path)
                else:
                    torch.save(save_dict, export_path + ".pt") 

                print("Exported!")
                modal.close()



    def paint_mask_points(self, render_rgbs, img_wh):
        print("Painting mask points")
        print(render_rgbs.shape)

        object_id = self.choose_points_object_id

        color = tuple(
            c / 255 for c in self.objects[object_id].get("indicator color", (255, 0, 0))
        )

        for point in self.objects[object_id].get("mask 2d points", []):
            x = int(point[0] * img_wh[0])
            y = int(point[1] * img_wh[1])
            size = 6  # Size of the X
            thickness = 2  # Thickness of the lines

            # Draw the two diagonal lines to form an X
            cv2.line(
                render_rgbs,
                (x - size, y - size),
                (x + size, y + size),
                color,
                thickness,
            )
            cv2.line(
                render_rgbs,
                (x - size, y + size),
                (x + size, y - size),
                color,
                thickness,
            )

        return render_rgbs

    def preprocess_click(self, _):
        self.sam_features = {}
        self.render_images = {}
        print("Prepocessing: extracting SAM features...")

        progress_max = len(self.cameras)
        self.preprocess_progress.visible = True

        for i, camera in tqdm(enumerate(self.cameras)):
            self.preprocess_progress.value = (i / progress_max) * 100
            # image_name = train
            render_pkg = self.model.get_outputs_for_camera(self.cameras[i : i + 1])

            render_image = render_pkg["rgb"].detach().cpu().numpy()
            render_image = (255 * np.clip(render_image, 0, 1)).astype(np.uint8)

            with torch.inference_mode() and torch.autocast(
                "cuda", dtype=torch.bfloat16
            ):
                self.predictor.set_image(render_image)
            self.sam_features[i] = self.predictor._features
        print("Preprocessing done.")
        self.preprocess_progress.visible = False

        self.processing_folder.remove()
        self.preprocessing_done = True

        # enable segment gaussians buttons
        for object_id in self.objects:
            if self.objects[object_id].get("3d prompts", None) is not None:
                self.objects[object_id]["segment_gaussians_btn"].disabled = False
                self.objects[object_id][
                    "segment_gaussians_btn"
                ].hint = "Segment 3D object from the scene"

    def client_cam_2_nerfview_cam(self, client_camera):
        wxyz = client_camera.wxyz
        position = client_camera.position

        rotation = Rotation.from_quat([wxyz[1], wxyz[2], wxyz[3], wxyz[0]]).as_matrix()
        c2w = np.hstack((rotation, position.reshape(3, 1)))
        c2w = np.vstack((c2w, np.array([0, 0, 0, 1])))

        client_cam_state = nerfview.CameraState(
            fov=client_camera.fov, aspect=client_camera.aspect, c2w=c2w
        )

        return client_cam_state

    def client_cam_2_nerfstudio_cam(self, client_camera):
        # client camera (origin)
        # self._state = _CameraHandleState(
        #    client,
        #    wxyz=np.zeros(4),
        #    position=np.zeros(3),
        #    fov=0.0,
        #    aspect=0.0,
        #    look_at=np.zeros(3),
        #    up_direction=np.zeros(3),
        #    update_timestamp=0.0,
        #    camera_cb=[],
        # )

        # nerfstudio camera (target)
        #    camera_to_worlds: Float[Tensor, "*num_cameras 3 4"]
        #    fx: Float[Tensor, "*num_cameras 1"]
        #    fy: Float[Tensor, "*num_cameras 1"]
        #    cx: Float[Tensor, "*num_cameras 1"]
        #    cy: Float[Tensor, "*num_cameras 1"]
        #    width: Shaped[Tensor, "*num_cameras 1"]
        #    height: Shaped[Tensor, "*num_cameras 1"]
        #    distortion_params: Optional[Float[Tensor, "*num_cameras 6"]]
        #    camera_type: Int[Tensor, "*num_cameras 1"]
        #    times: Optional[Float[Tensor, "num_cameras 1"]]
        #    metadata: Optional[Dict]

        wxyz = client_camera.wxyz
        position = client_camera.position

        rotation = Rotation.from_quat([wxyz[1], wxyz[2], wxyz[3], wxyz[0]]).as_matrix()
        c2w = np.hstack((rotation, position.reshape(3, 1)))

        c2w[:, 1] = -c2w[:, 1]
        c2w[:, 2] = -c2w[:, 2]

        aspect = client_camera.aspect
        width = 1000
        height = int(width / aspect)

        fov = client_camera.fov

        fy = height / (2 * math.tan(fov / 2))
        fx = fy  # width / (2 * math.tan(fov / 2))
        cx = width / 2
        cy = height / 2

        nerfstudio_cam = nerfCameras(
            camera_to_worlds=torch.from_numpy(c2w).float().unsqueeze(0),
            fx=torch.tensor(fx).unsqueeze(0),
            fy=torch.tensor(fy).unsqueeze(0),
            cx=torch.tensor(cx).unsqueeze(0),
            cy=torch.tensor(cy).unsqueeze(0),
            width=torch.tensor(width).unsqueeze(0),
            height=torch.tensor(height).unsqueeze(0),
            distortion_params=torch.tensor([0, 0, 0, 0, 0, 0]).unsqueeze(0),
            camera_type=torch.tensor(1).unsqueeze(0),
        )

        return nerfstudio_cam

    def nerfstudio_cam_2_nerfview_cam(self, nerfstudio_cam):
        c2w = nerfstudio_cam.camera_to_worlds.numpy()
        c2w = np.vstack((c2w, np.array([0, 0, 0, 1])))

        fx = nerfstudio_cam.fx[0].item()

        width = nerfstudio_cam.width[0].item()
        height = nerfstudio_cam.height[0].item()

        aspect = width / height
        fov = 2 * math.atan(width / (2 * fx))

        nerfview_cam = nerfview.CameraState(fov=fov, aspect=aspect, c2w=c2w)

        return nerfview_cam

    def compute_point_size(self, points, camera, size):
        camera_to_worlds = camera.camera_to_worlds
        fx = camera.fx
        fy = camera.fy

        # Ensure points has the correct dimensions
        points = torch.tensor(points, dtype=torch.float32)
        if points.ndim == 2:
            points = points.unsqueeze(0)  # Shape: [1, num_points, 3]

        # Decompose camera-to-world matrix
        R = camera_to_worlds[:, :3, :3]  # [num_cameras, 3, 3]
        t = camera_to_worlds[:, :3, 3]  # [num_cameras, 3]

        # Compute Euclidean distance from camera origin to points
        points_world = points - t.unsqueeze(1)  # [num_cameras, num_points, 3]
        distances = torch.linalg.norm(points_world, dim=-1)  # [num_cameras, num_points]

        # Average distance across all points
        avg_distance = distances.mean(dim=1)  # [num_cameras]

        # Compute size in world units based on average distance
        sizes_world = (size * avg_distance) / fx  # [num_cameras]

        return sizes_world

    def render_2d_changes(self, render_rgbs, img_wh):
        if self.show_mask_points:
            render_rgbs = self.paint_mask_points(render_rgbs, img_wh)
        # if self.display_style:
        #    render_rgbs = self.render_display_style(render_rgbs, img_wh)
        if self.display_style_progress:
            render_rgbs = self.render_display_style_progress(render_rgbs, img_wh)

        return render_rgbs

    def render_display_style(self, render_rgbs, img_wh):
        width, height = img_wh
        style_img = np.array(self.style_img)
        style_width, style_height = style_img.shape[1], style_img.shape[0]

        # resize so its maximum 1/6 of the images height
        max_height = height // 4
        max_width = width // 4
        # style_img = self.style_img.copy()

        padding = min(max_height, max_width) // 15

        if style_height > max_height or style_width > max_width:
            scale_factor = min(max_height / style_height, max_width / style_width)
            style_img = cv2.resize(style_img, (0, 0), fx=scale_factor, fy=scale_factor)

        style_width, style_height = style_img.shape[1], style_img.shape[0]

        # paste style image in the corner
        render_rgbs[
            height - style_height - padding : height - padding,
            width - style_width - padding : width - padding,
        ] = style_img[:, :, :3] / 255

        # render_rgbs[0:self.style_img.shape[0], 0:self.style_img.shape[1]] = self.style_img[:, :, :3] / 255

        return render_rgbs

    def render_display_style_progress(self, render_rgbs, img_wh):
        width, height = img_wh
        style_width, style_height = (
            self.style_progress.shape[1],
            self.style_progress.shape[0],
        )

        # resize so its maximum 1/6 of the images height
        max_height = height // 4
        max_width = width // 4
        style_img = self.style_progress.copy()

        padding = min(max_height, max_width) // 15

        if style_height > max_height or style_width > max_width:
            scale_factor = min(max_height / style_height, max_width / style_width)
            style_img = cv2.resize(style_img, (0, 0), fx=scale_factor, fy=scale_factor)

        style_width, style_height = style_img.shape[1], style_img.shape[0]

        # paste style image in the corner
        render_rgbs[
            height - style_height - padding : height - padding,
            width - style_width - padding : width - padding,
        ] = style_img[:, :, :3] / 255

        return render_rgbs

    # register and open viewer
    @torch.no_grad()
    def viewer_render_fn(
        self, camera_state: nerfview.CameraState, img_wh: Tuple[int, int]
    ):
        width, height = img_wh
        c2w = camera_state.c2w
        K = camera_state.get_K(img_wh)
        c2w = torch.from_numpy(c2w).float().to(device)
        K = torch.from_numpy(K).float().to(device)
        viewmat = c2w.inverse()

        if args.backend == "gsplat":
            rasterization_fn = rasterization
        elif args.backend == "gsplat_legacy":
            from gsplat import rasterization_legacy_wrapper

            rasterization_fn = rasterization_legacy_wrapper
        elif args.backend == "inria":
            from gsplat import rasterization_inria_wrapper

            rasterization_fn = rasterization_inria_wrapper
        else:
            raise ValueError

        render_colors = None

        colors = self.colors.clone()

        for obj in self.objects.values():
            if obj.get("matched colors", None) is not None:
                colors[obj["final mask"]] = (
                    torch.sigmoid(obj["matched colors"]).squeeze(-1).detach()
                )

            if obj.get("styled features dc", None) is not None:
                colors[obj["final mask"]] = (
                    torch.sigmoid(obj["styled features dc"]).squeeze(-1).detach()
                )

            if obj.get("highlight mask", False):
                mask = obj["final mask"]
                color = obj.get("indicator color", (255, 0, 0))
                color = tuple(c / 255 for c in color)

                colors[mask] = (
                    colors[mask] * 0.5 + torch.tensor(color, device=device) * 0.5
                )
        render_colors, render_alphas, meta = rasterization_fn(
            self.means,  # [N, 3]
            self.quats,  # [N, 4]
            self.scales,  # [N, 3]
            self.opacities,  # [N]
            colors,  # [N, 3]
            viewmat[None],  # [1, 4, 4]
            K[None],  # [1, 3, 3]
            width,
            height,
            sh_degree=self.sh_degree,
            render_mode="RGB",
            backgrounds=torch.ones(1, 3, device=device),
            # this is to speedup large-scale rendering by skipping far-away Gaussians.
            # radius_clip=3,
        )

        render_rgbs = render_colors[0, ..., 0:3].cpu().numpy()

        render_rgbs = self.render_2d_changes(render_rgbs, img_wh)
        # if self.save_render_img:
        #    self.mask_img = (render_rgbs.copy() * 255).astype(np.uint8)

        return render_rgbs


def main():
    viewer()


if __name__ == "__main__":
    main()
