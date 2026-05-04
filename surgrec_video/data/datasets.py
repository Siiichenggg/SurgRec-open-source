import os
from surgrec_video.augment.transforms import *
from surgrec_video.data.kinetics import VideoClsDataset
from surgrec_video.data.ssv2 import SSVideoClsDataset


def build_dataset(is_train, test_mode, args):
    if args.data_set == 'Kinetics-400':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv') 

        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 400
    
    elif args.data_set == 'SSV2':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv') 

        dataset = SSVideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=1,
            num_segment=args.num_frames,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 174

    elif args.data_set == 'UCF101':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv') 

        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 101
    
    elif args.data_set == 'HMDB51':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv') 

        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 51
    elif args.data_set == 'image_folder':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv')
        else:
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'test.csv')
        dataset = VideoClsDataset(
                anno_path=anno_path,
                data_path='',
                mode=mode,
                clip_len=args.num_frames,
                frame_sample_rate=args.sampling_rate,
                num_segment=1,
                test_num_segment=args.test_num_segment,
                test_num_crop=args.test_num_crop,
                num_crop=1 if not test_mode else 3,
                keep_aspect_ratio=True,
                crop_size=args.input_size,
                short_side_size=args.short_side_size,
                new_height=224,
                new_width=224,
                args=args)
        nb_classes = 2
    elif args.data_set == 'SurgKinetics':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train_clean.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test_clean.csv')
        else:
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'test_clean.csv')
        dataset = VideoClsDataset(
                anno_path=anno_path,
                data_path='',
                mode=mode,
                clip_len=args.num_frames,
                frame_sample_rate=args.sampling_rate,
                num_segment=1,
                test_num_segment=args.test_num_segment,
                test_num_crop=args.test_num_crop,
                num_crop=1 if not test_mode else 3,
                keep_aspect_ratio=True,
                crop_size=args.input_size,
                short_side_size=args.short_side_size,
                new_height=224,
                new_width=224,
                args=args)
        nb_classes = 150
    elif args.data_set == 'colonoscopic_web':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv')
        else:
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'test.csv')
        dataset = VideoClsDataset(
                anno_path=anno_path,
                data_path='',
                mode=mode,
                clip_len=args.num_frames,
                frame_sample_rate=args.sampling_rate,
                num_segment=1,
                test_num_segment=args.test_num_segment,
                test_num_crop=args.test_num_crop,
                num_crop=1 if not test_mode else 3,
                keep_aspect_ratio=True,
                crop_size=args.input_size,
                short_side_size=args.short_side_size,
                new_height=224,
                new_width=224,
                args=args)
        nb_classes = 3
    elif args.data_set == 'endovis2019':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv')
        else:
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'test.csv')
        dataset = VideoClsDataset(
                anno_path=anno_path,
                data_path='',
                mode=mode,
                clip_len=args.num_frames,
                frame_sample_rate=args.sampling_rate,
                num_segment=1,
                test_num_segment=args.test_num_segment,
                test_num_crop=args.test_num_crop,
                num_crop=1 if not test_mode else 3,
                keep_aspect_ratio=True,
                crop_size=args.input_size,
                short_side_size=args.short_side_size,
                new_height=224,
                new_width=224,
                args=args)
        nb_classes = 17
    elif args.data_set == 'JIGSAWS':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv')
        else:
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'test.csv')
        dataset = VideoClsDataset(
                anno_path=anno_path,
                data_path='',
                mode=mode,
                clip_len=args.num_frames,
                frame_sample_rate=args.sampling_rate,
                num_segment=1,
                test_num_segment=args.test_num_segment,
                test_num_crop=args.test_num_crop,
                num_crop=1 if not test_mode else 3,
                keep_aspect_ratio=True,
                crop_size=args.input_size,
                short_side_size=args.short_side_size,
                new_height=224,
                new_width=224,
                args=args)
        nb_classes = 13
    elif args.data_set == 'cholecT50':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv')
        else:
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'test.csv')
        dataset = VideoClsDataset(
                anno_path=anno_path,
                data_path='',
                mode=mode,
                clip_len=args.num_frames,
                frame_sample_rate=args.sampling_rate,
                num_segment=1,
                test_num_segment=args.test_num_segment,
                test_num_crop=args.test_num_crop,
                num_crop=1 if not test_mode else 3,
                keep_aspect_ratio=True,
                crop_size=args.input_size,
                short_side_size=args.short_side_size,
                new_height=224,
                new_width=224,
                args=args)
        nb_classes = 11
    elif args.data_set == 'Hyper-kvasir':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv')
        else:
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'test.csv')
        dataset = VideoClsDataset(
                anno_path=anno_path,
                data_path='',
                mode=mode,
                clip_len=args.num_frames,
                frame_sample_rate=args.sampling_rate,
                num_segment=1,
                test_num_segment=args.test_num_segment,
                test_num_crop=args.test_num_crop,
                num_crop=1 if not test_mode else 3,
                keep_aspect_ratio=True,
                crop_size=args.input_size,
                short_side_size=args.short_side_size,
                new_height=224,
                new_width=224,
                args=args)
        nb_classes = 9
    elif args.data_set == 'cholec80':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv')
        else:
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'test.csv')
        dataset = VideoClsDataset(
                anno_path=anno_path,
                data_path='',
                mode=mode,
                clip_len=args.num_frames,
                frame_sample_rate=args.sampling_rate,
                num_segment=1,
                test_num_segment=args.test_num_segment,
                test_num_crop=args.test_num_crop,
                num_crop=1 if not test_mode else 3,
                keep_aspect_ratio=True,
                crop_size=args.input_size,
                short_side_size=args.short_side_size,
                new_height=224,
                new_width=224,
                args=args)
        nb_classes = 7
    elif args.data_set == 'zju_phase':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv')
        else:
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'test.csv')
        dataset = VideoClsDataset(
                anno_path=anno_path,
                data_path='',
                mode=mode,
                clip_len=args.num_frames,
                frame_sample_rate=args.sampling_rate,
                num_segment=1,
                test_num_segment=args.test_num_segment,
                test_num_crop=args.test_num_crop,
                num_crop=1 if not test_mode else 3,
                keep_aspect_ratio=True,
                crop_size=args.input_size,
                short_side_size=args.short_side_size,
                new_height=224,
                new_width=224,
                args=args)
        nb_classes = 6
    elif args.data_set == 'AutoLaparo':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv')
        else:
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'test.csv')
        dataset = VideoClsDataset(
                anno_path=anno_path,
                data_path='',
                mode=mode,
                clip_len=args.num_frames,
                frame_sample_rate=args.sampling_rate,
                num_segment=1,
                test_num_segment=args.test_num_segment,
                test_num_crop=args.test_num_crop,
                num_crop=1 if not test_mode else 3,
                keep_aspect_ratio=True,
                crop_size=args.input_size,
                short_side_size=args.short_side_size,
                new_height=224,
                new_width=224,
                args=args)
        nb_classes = 14
    elif args.data_set == 'M2CAI16-Workflow':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv')
        else:
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'test.csv')
        dataset = VideoClsDataset(
                anno_path=anno_path,
                data_path='',
                mode=mode,
                clip_len=args.num_frames,
                frame_sample_rate=args.sampling_rate,
                num_segment=1,
                test_num_segment=args.test_num_segment,
                test_num_crop=args.test_num_crop,
                num_crop=1 if not test_mode else 3,
                keep_aspect_ratio=True,
                crop_size=args.input_size,
                short_side_size=args.short_side_size,
                new_height=224,
                new_width=224,
                args=args)
        nb_classes = 9
    elif args.data_set == 'SAR-RARP50':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv')
        else:
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'test.csv')
        dataset = VideoClsDataset(
                anno_path=anno_path,
                data_path='',
                mode=mode,
                clip_len=args.num_frames,
                frame_sample_rate=args.sampling_rate,
                num_segment=1,
                test_num_segment=args.test_num_segment,
                test_num_crop=args.test_num_crop,
                num_crop=1 if not test_mode else 3,
                keep_aspect_ratio=True,
                crop_size=args.input_size,
                short_side_size=args.short_side_size,
                new_height=224,
                new_width=224,
                args=args)
        nb_classes = 8
    elif args.data_set == 'PitVis':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv')
        else:
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'test.csv')
        dataset = VideoClsDataset(
                anno_path=anno_path,
                data_path='',
                mode=mode,
                clip_len=args.num_frames,
                frame_sample_rate=args.sampling_rate,
                num_segment=1,
                test_num_segment=args.test_num_segment,
                test_num_crop=args.test_num_crop,
                num_crop=1 if not test_mode else 3,
                keep_aspect_ratio=True,
                crop_size=args.input_size,
                short_side_size=args.short_side_size,
                new_height=224,
                new_width=224,
                args=args)
        nb_classes = 14
    elif args.data_set == 'LapGyn_dataset':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv')
        else:
            mode = 'validation'
            val_path = os.path.join(args.data_path, 'val.csv')
            anno_path = val_path if os.path.isfile(val_path) else os.path.join(args.data_path, 'test.csv')
        dataset = VideoClsDataset(
                anno_path=anno_path,
                data_path='',
                mode=mode,
                clip_len=args.num_frames,
                frame_sample_rate=args.sampling_rate,
                num_segment=1,
                test_num_segment=args.test_num_segment,
                test_num_crop=args.test_num_crop,
                num_crop=1 if not test_mode else 3,
                keep_aspect_ratio=True,
                crop_size=args.input_size,
                short_side_size=args.short_side_size,
                new_height=224,
                new_width=224,
                args=args)
        def infer_num_classes(path):
            max_label = -1
            if not os.path.isfile(path):
                return max_label
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                for line in f:
                    s = line.strip()
                    if not s or s.startswith('#'):
                        continue
                    parts = s.replace(',', ' ').split()
                    if not parts:
                        continue
                    try:
                        label = int(parts[-1])
                    except Exception:
                        continue
                    max_label = max(max_label, label)
            return max_label + 1 if max_label >= 0 else max_label
        inferred = infer_num_classes(os.path.join(args.data_path, 'train.csv'))
        if inferred <= 0:
            inferred = infer_num_classes(os.path.join(args.data_path, 'test.csv'))
        nb_classes = inferred if inferred > 0 else args.nb_classes
    elif args.data_set == 'MultiBypass140':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv')
        else:
            mode = 'validation'
            val_path = os.path.join(args.data_path, 'val.csv')
            anno_path = val_path if os.path.isfile(val_path) else os.path.join(args.data_path, 'test.csv')

        dataset = VideoClsDataset(
                anno_path=anno_path,
                data_path='',
                mode=mode,
                clip_len=args.num_frames,
                frame_sample_rate=args.sampling_rate,
                num_segment=1,
                test_num_segment=args.test_num_segment,
                test_num_crop=args.test_num_crop,
                num_crop=1 if not test_mode else 3,
                keep_aspect_ratio=True,
                crop_size=args.input_size,
                short_side_size=args.short_side_size,
                new_height=224,
                new_width=224,
                args=args)

        def infer_num_classes(path):
            max_label = -1
            if not os.path.isfile(path):
                return max_label
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    s = line.strip()
                    if not s or s.startswith('#'):
                        continue
                    if ',' in s:
                        parts = s.split(',')
                        label_token = parts[-1].strip()
                    else:
                        parts = s.split()
                        label_token = parts[-1].strip() if parts else None
                    if not label_token:
                        continue
                    try:
                        label = int(label_token)
                    except Exception:
                        continue
                    if label > max_label:
                        max_label = label
            return max_label + 1 if max_label >= 0 else max_label

        inferred = infer_num_classes(os.path.join(args.data_path, 'train.csv'))
        if inferred <= 0:
            inferred = infer_num_classes(os.path.join(args.data_path, 'test.csv'))
        nb_classes = inferred if inferred > 0 else args.nb_classes
    elif args.data_set == 'SAR-RARP50':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv')
        else:
            mode = 'validation'
            val_path = os.path.join(args.data_path, 'val.csv')
            anno_path = val_path if os.path.isfile(val_path) else os.path.join(args.data_path, 'test.csv')

        dataset = VideoClsDataset(
                anno_path=anno_path,
                data_path='',
                mode=mode,
                clip_len=args.num_frames,
                frame_sample_rate=args.sampling_rate,
                num_segment=1,
                test_num_segment=args.test_num_segment,
                test_num_crop=args.test_num_crop,
                num_crop=1 if not test_mode else 3,
                keep_aspect_ratio=True,
                crop_size=args.input_size,
                short_side_size=args.short_side_size,
                new_height=224,
                new_width=224,
                args=args)

        def infer_num_classes(path):
            max_label = -1
            if not os.path.isfile(path):
                return max_label
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    s = line.strip()
                    if not s or s.startswith('#'):
                        continue
                    if ',' in s:
                        parts = s.split(',')
                        label_token = parts[-1].strip()
                    else:
                        parts = s.split()
                        label_token = parts[-1].strip() if parts else None
                    if not label_token:
                        continue
                    try:
                        label = int(label_token)
                    except Exception:
                        continue
                    if label > max_label:
                        max_label = label
            return max_label + 1 if max_label >= 0 else max_label

        inferred = infer_num_classes(os.path.join(args.data_path, 'train.csv'))
        if inferred <= 0:
            inferred = infer_num_classes(os.path.join(args.data_path, 'test.csv'))
        nb_classes = inferred if inferred > 0 else args.nb_classes
    elif args.data_set == 'LDPolyVideo':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv')
        else:
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'test.csv')
        dataset = VideoClsDataset(
                anno_path=anno_path,
                data_path='',
                mode=mode,
                clip_len=args.num_frames,
                frame_sample_rate=args.sampling_rate,
                num_segment=1,
                test_num_segment=args.test_num_segment,
                test_num_crop=args.test_num_crop,
                num_crop=1 if not test_mode else 3,
                keep_aspect_ratio=True,
                crop_size=args.input_size,
                short_side_size=args.short_side_size,
                new_height=224,
                new_width=224,
                args=args)
        nb_classes = 2
    elif args.data_set == 'AlxSuture':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv')
        else:
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv')

        dataset = VideoClsDataset(
                anno_path=anno_path,
                data_path='',
                mode=mode,
                clip_len=args.num_frames,
                frame_sample_rate=args.sampling_rate,
                num_segment=1,
                test_num_segment=args.test_num_segment,
                test_num_crop=args.test_num_crop,
                num_crop=1 if not test_mode else 3,
                keep_aspect_ratio=True,
                crop_size=args.input_size,
                short_side_size=args.short_side_size,
                new_height=224,
                new_width=224,
                args=args)
        nb_classes = 3
    elif args.data_set == 'SurgicalActions160':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv')
        else:
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv')
        dataset = VideoClsDataset(
                anno_path=anno_path,
                data_path='',
                mode=mode,
                clip_len=args.num_frames,
                frame_sample_rate=args.sampling_rate,
                num_segment=1,
                test_num_segment=args.test_num_segment,
                test_num_crop=args.test_num_crop,
                num_crop=1 if not test_mode else 3,
                keep_aspect_ratio=True,
                crop_size=args.input_size,
                short_side_size=args.short_side_size,
                new_height=224,
                new_width=224,
                args=args)
        nb_classes = args.nb_classes
    elif args.data_set == 'PitVis':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv')
        else:
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv')
        dataset = VideoClsDataset(
                anno_path=anno_path,
                data_path='',
                mode=mode,
                clip_len=args.num_frames,
                frame_sample_rate=args.sampling_rate,
                num_segment=1,
                test_num_segment=args.test_num_segment,
                test_num_crop=args.test_num_crop,
                num_crop=1 if not test_mode else 3,
                keep_aspect_ratio=True,
                crop_size=args.input_size,
                short_side_size=args.short_side_size,
                new_height=224,
                new_width=224,
                args=args)
        nb_classes = args.nb_classes
    elif args.data_set == 'all':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv')
        else:
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'test.csv')
        dataset = VideoClsDataset(
                anno_path=anno_path,
                data_path='',
                mode=mode,
                clip_len=args.num_frames,
                frame_sample_rate=args.sampling_rate,
                num_segment=1,
                test_num_segment=args.test_num_segment,
                test_num_crop=args.test_num_crop,
                num_crop=1 if not test_mode else 3,
                keep_aspect_ratio=True,
                crop_size=args.input_size,
                short_side_size=args.short_side_size,
                new_height=224,
                new_width=224,
                args=args)
        nb_classes = 72
    elif args.data_set == 'OOD':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv')
        else:
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'test.csv')
        dataset = VideoClsDataset(
                anno_path=anno_path,
                data_path='',
                mode=mode,
                clip_len=args.num_frames,
                frame_sample_rate=args.sampling_rate,
                num_segment=1,
                test_num_segment=args.test_num_segment,
                test_num_crop=args.test_num_crop,
                num_crop=1 if not test_mode else 3,
                keep_aspect_ratio=True,
                crop_size=args.input_size,
                short_side_size=args.short_side_size,
                new_height=224,
                new_width=224,
                args=args)
        nb_classes = 4
    elif args.data_set == 'kvasir-capsule':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv')
        else:
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'test.csv')
        dataset = VideoClsDataset(
                anno_path=anno_path,
                data_path='',
                mode=mode,
                clip_len=args.num_frames,
                frame_sample_rate=args.sampling_rate,
                num_segment=1,
                test_num_segment=args.test_num_segment,
                test_num_crop=args.test_num_crop,
                num_crop=1 if not test_mode else 3,
                keep_aspect_ratio=True,
                crop_size=args.input_size,
                short_side_size=args.short_side_size,
                new_height=224,
                new_width=224,
                args=args)
        nb_classes = 7
    elif args.data_set == 'OOD_cataract-101':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv')
        else:
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'test.csv')
        dataset = VideoClsDataset(
                anno_path=anno_path,
                data_path='',
                mode=mode,
                clip_len=args.num_frames,
                frame_sample_rate=args.sampling_rate,
                num_segment=1,
                test_num_segment=args.test_num_segment,
                test_num_crop=args.test_num_crop,
                num_crop=1 if not test_mode else 3,
                keep_aspect_ratio=True,
                crop_size=args.input_size,
                short_side_size=args.short_side_size,
                new_height=224,
                new_width=224,
                args=args)
        nb_classes = 10
    elif args.data_set == 'OOD_cat-21':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv')
        else:
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'test.csv')
        dataset = VideoClsDataset(
                anno_path=anno_path,
                data_path='',
                mode=mode,
                clip_len=args.num_frames,
                frame_sample_rate=args.sampling_rate,
                num_segment=1,
                test_num_segment=args.test_num_segment,
                test_num_crop=args.test_num_crop,
                num_crop=1 if not test_mode else 3,
                keep_aspect_ratio=True,
                crop_size=args.input_size,
                short_side_size=args.short_side_size,
                new_height=224,
                new_width=224,
                args=args)
        nb_classes = 12
    elif args.data_set == 'OOD_cataract-1k':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv')
        else:
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'test.csv')
        dataset = VideoClsDataset(
                anno_path=anno_path,
                data_path='',
                mode=mode,
                clip_len=args.num_frames,
                frame_sample_rate=args.sampling_rate,
                num_segment=1,
                test_num_segment=args.test_num_segment,
                test_num_crop=args.test_num_crop,
                num_crop=1 if not test_mode else 3,
                keep_aspect_ratio=True,
                crop_size=args.input_size,
                short_side_size=args.short_side_size,
                new_height=224,
                new_width=224,
                args=args)
        nb_classes = 12
    else:
    
        raise NotImplementedError()
    args.nb_classes = nb_classes
    # assert nb_classes == args.nb_classes
    print("Number of the class = %d" % args.nb_classes)

    return dataset, nb_classes
