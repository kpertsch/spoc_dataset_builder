from typing import Iterator, Tuple, Any

import cv2
import glob
import h5py
import os
import json
import numpy as np
import tensorflow_datasets as tfds
from spoc.conversion_utils import MultiThreadedDatasetBuilder, parse_bbox, convert_byte_to_string


def _generate_examples(paths) -> Iterator[Tuple[str, Any]]:
    """Yields episodes for list of data paths."""

    def _parse_example(episode_path):
        # load raw data
        try:
            with h5py.File(episode_path, "r") as F:
                an_object_is_in_hand = F['0']['an_object_is_in_hand'][()]
                house_index = F['0']['house_index'][()]
                hypothetical_task_success = F['0']['hypothetical_task_success'][()]
                last_action_is_random = F['0']['last_action_is_random'][()]
                last_action_str = F['0']['last_action_str'][()]
                last_action_success = F['0']['last_action_success'][()]
                last_agent_location = F['0']['last_agent_location'][()]
                minimum_l2_target_distance = F['0']['minimum_l2_target_distance'][()]
                minimum_visible_target_alignment = F['0']['minimum_visible_target_alignment'][()]
                relative_arm_location_metadata = F['0']['relative_arm_location_metadata'][()]
                room_current_seen = F['0']['room_current_seen'][()]
                rooms_seen = F['0']['rooms_seen'][()]
                templated_task_spec = F['0']['templated_task_spec'][()]
                visible_target_4m_count = F['0']['visible_target_4m_count'][()]

                nav_object_bbox_group = F['0']['nav_accurate_object_bbox']
                nav_bbox = {name: value[()] if isinstance(value, h5py.Dataset) else dict(value) for name, value in
                            nav_object_bbox_group.items()}
                manip_object_bbox_group = F['0']['manip_accurate_object_bbox']
                manip_bbox = {name: value[()] if isinstance(value, h5py.Dataset) else dict(value) for name, value in
                              manip_object_bbox_group.items()}


        except:
            print(f"Could not extract data for {episode_path} -- skipping.")
            return None

        # extract language instruction
        try:
            task_dict = json.loads(bytearray(list(templated_task_spec[-1])).decode("utf-8"))
            language_instruction = task_dict['extras']['natural_language_description']
        except:
            print(f"Failed to extract language instruction for {episode_path} -- skipping")
            return None

        # get the bounding boxes
        # no receptacle IDs in this dataset, just the targets
        tgt_1_ids = []
        if "broad_synset_to_object_ids" in task_dict:
            tgt_1_ids = [
                val for val in task_dict["broad_synset_to_object_ids"].values()
            ]
            tgt_1_ids = sum(tgt_1_ids, [])
        nav_object_bbox = parse_bbox(nav_bbox, tgt_1_ids)
        manip_object_bbox = parse_bbox(manip_bbox, tgt_1_ids)


        # extract video frames
        def get_frames(filepath):
            vid = cv2.VideoCapture(filepath)
            vid_frames = []
            while True:
                success, frame = vid.read()
                if not success:
                    break
                vid_frames.append(frame[..., ::-1])
            return vid_frames

        try:
            nav_cam_frames = get_frames(
                os.path.join(os.path.dirname(episode_path), "raw_navigation_camera__0.mp4"))
            manipulation_cam_frames = get_frames(
                os.path.join(os.path.dirname(episode_path), "raw_manipulation_camera__0.mp4"))
            if len(nav_cam_frames) != len(manipulation_cam_frames):
                print(f"Number of frames does not match for {episode_path}-- skipping.")
                return None
        except:
            print(f"Failed to extract video frames for {episode_path} -- skipping")
            return None

        dimensions = ["base_theta", "base_z", "arm_y", "arm_z", "wrist_yaw", "pickup", "dropoff", "done", "sub_done"]
        try:
            act, prev_act_str = [], []
            raw_prev_action_strings = [convert_byte_to_string(a) for a in last_action_str]
            for idx in range(len(raw_prev_action_strings)-1):
                # Parse JSON or handle non-dict actions
                if raw_prev_action_strings[idx+1].startswith('{'):
                    act_dict = json.loads(raw_prev_action_strings[idx+1])
                    action = np.zeros((len(dimensions),), dtype=np.float32)

                    if "theta" in act_dict.get("action_values", {}).get("base", {}):
                        action[dimensions.index("base_theta")] = act_dict["action_values"]["base"]["theta"]
                    if "z" in act_dict.get("action_values", {}).get("base", {}):
                        action[dimensions.index("base_z")] = act_dict["action_values"]["base"]["z"]
                    if "y" in act_dict.get("action_values", {}).get("arm", {}):
                        action[dimensions.index("arm_y")] = act_dict["action_values"]["arm"]["y"]
                    if "z" in act_dict.get("action_values", {}).get("arm", {}):
                        action[dimensions.index("arm_z")] = act_dict["action_values"]["arm"]["z"]
                    if "yaw" in act_dict.get("action_values", {}).get("wrist", {}):
                        action[dimensions.index("wrist_yaw")] = act_dict["action_values"]["wrist"]["yaw"]
                else:
                    # Handle special actions
                    action = np.zeros((len(dimensions),), dtype=np.float32)
                    action[dimensions.index(raw_prev_action_strings[idx+1])] = 1.0

                act.append(action)
                prev_act_str.append(raw_prev_action_strings[idx])
            # NOTE: as written done does not appear in the prev_act_str. It is always the last action but that has
            # to stay implicit for the dimensions to work out.
        except:
            print(f"Failed to extract actions for {episode_path} -- skipping")
            return None

        # assemble episode --> here we're assuming demos so we set reward to 1 at the end
        try:
            episode = []
            for i in range(len(nav_cam_frames) - 1):
                episode.append({
                    'observation': dict(
                        image=nav_cam_frames[i],
                        image_manipulation=manipulation_cam_frames[i],
                        an_object_is_in_hand=bool(an_object_is_in_hand[i]),
                        house_index=house_index[i],
                        hypothetical_task_success=bool(hypothetical_task_success[i]),
                        last_action_is_random=bool(last_action_is_random[i]),
                        last_action_str=prev_act_str[i],
                        last_action_success=bool(last_action_success[i]),
                        last_agent_location=np.asarray(last_agent_location[i], dtype=np.float32),
                        manip_object_bbox=manip_object_bbox[i],
                        minimum_l2_target_distance=np.asarray(minimum_l2_target_distance[i][0], dtype=np.float32),
                        minimum_visible_target_alignment=np.asarray(minimum_visible_target_alignment[i][0], dtype=np.float32),
                        nav_object_bbox=nav_object_bbox[i],
                        relative_arm_location_metadata=np.asarray(relative_arm_location_metadata[i], dtype=np.float32),
                        room_current_seen=bool(room_current_seen[i]),
                        rooms_seen=rooms_seen[i],
                        visible_target_4m_count=visible_target_4m_count[i][0],
                    ),
                    'action': act[i],
                    'discount': 1.0,
                    'reward': float(i == (len(nav_cam_frames) - 1)),
                    'is_first': i == 0,
                    'is_last': i == (len(nav_cam_frames) - 1),
                    'is_terminal': i == (len(nav_cam_frames) - 1),
                    'language_instruction': language_instruction,
                })
        except:
            print(f"Failed to package output episode for {episode_path} -- skipping")
            return None

        # create output data sample
        sample = {
            'steps': episode,
            'episode_metadata': {
                'file_path': episode_path
            }
        }

        # if you want to skip an example for whatever reason, simply return None
        return episode_path, sample

    # for smallish datasets, use single-thread parsing
    for sample in paths:
        yield _parse_example(sample)


class Spoc(MultiThreadedDatasetBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }
    N_WORKERS = 10             # number of parallel workers for data conversion
    MAX_PATHS_IN_MEMORY = 100  # number of paths converted & stored in memory before writing to disk
                               # -> the higher the faster / more parallel conversion, adjust based on avilable RAM
                               # note that one path may yield multiple episodes and adjust accordingly
    PARSE_FCN = _generate_examples      # handle to parse function from file paths to RLDS episodes

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(224, 384, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Main camera RGB observation for navigation.',
                        ),
                        'image_manipulation': tfds.features.Image(
                            shape=(224, 384, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Wrist camera RGB observation for manipulation.',
                        ),
                        'an_object_is_in_hand': tfds.features.Scalar(
                            dtype=np.bool_,
                            doc='Indicates whether an object is currently being held in hand.',
                        ),
                        'house_index': tfds.features.Scalar(
                            dtype=np.int_,
                            doc='Indicates the index of the house in the dataset.',
                        ),
                        'hypothetical_task_success': tfds.features.Scalar(
                            dtype=np.bool_,
                            doc='Indicates whether the task will be successful if the agent '
                                'would have issued done at this step.',
                        ),
                        'last_action_is_random': tfds.features.Scalar(
                            dtype=np.bool_,
                            doc='Indicates if the most recent action taken was chosen randomly.',
                        ),
                        'last_action_str': tfds.features.Text(
                            doc="A string representation of the action previous to the current frame."
                        ),
                        'last_action_success': tfds.features.Scalar(
                            dtype=np.bool_,
                            doc='Indicates whether the last action performed was successful.',
                        ),
                        'last_agent_location': tfds.features.Tensor(
                            shape=(6,),
                            dtype=np.float32,
                            doc="Indicates agent's location in the world coordinate frame. [XYZ, 3xeuler in degree]",
                        ),
                        'manip_object_bbox': tfds.features.Tensor(
                            shape=(5,),
                            dtype=np.float32,
                            doc='Indicates the bounding box for the target object in the manipulation camera. '
                                'values are [x1, y1, x2, y2, area]. [1000, 1000, 1000, 1000, 0] for no box.',
                        ),
                        'minimum_l2_target_distance': tfds.features.Scalar(
                            dtype=np.float32,
                            doc='The minimum Euclidean (L2) distance to the target object or location.',
                        ),
                        'minimum_visible_target_alignment': tfds.features.Scalar(
                            dtype=np.float32,
                            doc='Measures the minimum degree the agent needs to turn to center the object '
                                'in the navigation camera frame (if object is visible).',
                        ),
                        'nav_object_bbox': tfds.features.Tensor(
                            shape=(5,),
                            dtype=np.float32,
                            doc='Indicates the bounding box for the target object in the navigation camera. '
                                'values are [x1, y1, x2, y2, area]. [1000, 1000, 1000, 1000, 0] for no box.',
                        ),
                        'relative_arm_location_metadata': tfds.features.Tensor(
                            shape=(4,),
                            dtype=np.float32,
                            doc="Arm proprioceptive, relative location of the wrist in the agent's coordinate frame "
                                "[x,y,z,wrist yaw in degrees]",
                        ),
                        'room_current_seen': tfds.features.Scalar(
                            dtype=np.bool_,
                            doc='Indicates whether this room has been seen before or not.',
                        ),
                        'rooms_seen': tfds.features.Scalar(
                            dtype=np.int_,
                            doc='Count of rooms that have been visited by the agent.',
                        ),
                        'visible_target_4m_count': tfds.features.Scalar(
                            dtype=np.int_,
                            doc='The count of targets visible within a 4-meter radius or distance.',
                        ),
                    }),
                    'action': tfds.features.Tensor(
                        shape=(9,),
                        dtype=np.float32,
                        doc='Spatial and special robot actions. Dimensions are '
                            '["base_theta", "base_z", "arm_y", "arm_z", "wrist_yaw", "pickup", "dropoff", "done", "sub_done"]'
                            'the last four are binary flags for the respective special actions, '
                            '["base_z", "arm_y", "arm_z"] translation in meters, '
                            '["base_theta","wrist_yaw"] rotation in degrees.'
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _split_paths(self):
        """Define filepaths for data splits."""
        print(self.info)
        return {
            'train': glob.glob('/Users/roseh/Downloads/quantized_ObjectNavOpenVocab/train/*/hdf5_sensors.hdf5'),
        }

