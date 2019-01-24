"""Problem Set 7: Particle Filter Tracking"""

from ps7 import *

# I/O directories
input_dir = "input"
output_dir = "output"

import matplotlib.pyplot as plt




frame_num = 0
print frame_num
video_filename = os.path.join(input_dir, "pres_debate.mp4")
video = cv2.VideoCapture(video_filename)



# Driver/helper code
def run_particle_filter(pf_class, video_filename, template_rect, save_frames={}, **kwargs):
    """Instantiates and runs a particle filter on a given video and template.

    Create an object of type pf_class, passing in initial video frame, template (extracted from first frame using
    template_rect), and any keyword arguments.

    Do not modify this function except for the debugging flag used to display every frame.

    Args:
        pf_class (object): particle filter class to instantiate (e.g. ParticleFilter).
        video_filename (str): path to input video file.
        template_rect (dict): template bounds (x, y, w, h), as float or int.
        save_frames (dict): frames to save {<frame number>|'template': <filename>}.
        **kwargs: arbitrary keyword arguments passed on to particle filter class.

    Returns:
        None.

    """

    # Open video file

    video = cv2.VideoCapture(video_filename)

    # Initialize objects
    template = None
    pf = None
    frame_num = 0

    # Loop over video (until last frame or Ctrl + C is presssed)
    while True:
        try:
            # Try to read a frame
            okay, frame = video.read()
            if not okay:
                break  # no more frames, or can't read video

            # Extract template and initialize (one-time only)
            if template is None:
                template = frame[int(template_rect['y']):int(template_rect['y'] + template_rect['h']),
                                 int(template_rect['x']):int(template_rect['x'] + template_rect['w'])]
                if 'template' in save_frames:
                    cv2.imwrite(save_frames['template'], template)
                pf = pf_class(frame, template, **kwargs)

            # Process frame
            pf.process(frame)  
            
            if False:  # For debugging, it displays every frame
                out_frame = frame.copy()
                pf.render(out_frame)
                cv2.imshow('Tracking', out_frame)
                cv2.waitKey(1)

            # Render and save output, if indicated
            if frame_num in save_frames:
                frame_out = frame.copy()
                pf.render(frame_out)
                cv2.imwrite(save_frames[frame_num], frame_out)
#            print frame_num
            # Update frame number
            frame_num += 1

        except KeyboardInterrupt:  # press ^C to quit
            break


def part_1a():
    num_particles = 1000  # Define the number of particles
    sigma_exp = 10.  # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 10.  # Define the value of sigma for the particles movement (dynamics)
    template_rect = {'x': 320.8751, 'y': 175.1776, 'w': 103.5404, 'h': 129.0504}  # suggested template window (dict)

    run_particle_filter(ParticleFilter,  # particle filter model class
                        os.path.join(input_dir, "pres_debate.mp4"),  # input video
                        template_rect,
                        {
                            'template': os.path.join(output_dir, 'ps7-1-a-1.png'),
                            28: os.path.join(output_dir, 'ps7-1-a-2.png'),
                            94: os.path.join(output_dir, 'ps7-1-a-3.png'),
                            171: os.path.join(output_dir, 'ps7-1-a-4.png')
                        },  # frames to save, mapped to filenames, and 'template' if desired
                        num_particles=num_particles, sigma_exp=sigma_exp, sigma_dyn=sigma_dyn,
                        template_coords=template_rect)  # Add more if you need to


def part_1b():
    num_particles = 1000  # Define the number of particles
    sigma_exp = 10.  # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 10.  # Define the value of sigma for the particles movement (dynamics)
    template_rect = {'x': 320.8751, 'y': 175.1776, 'w': 103.5404, 'h': 129.0504}

    run_particle_filter(ParticleFilter,
                        os.path.join(input_dir, "noisy_debate.mp4"),
                        template_rect,
                        {
                            14: os.path.join(output_dir, 'ps7-1-b-1.png'),
                            94: os.path.join(output_dir, 'ps7-1-b-2.png'),
                            530: os.path.join(output_dir, 'ps7-1-b-3.png')
                        },
                        num_particles=num_particles, sigma_exp=sigma_exp, sigma_dyn=sigma_dyn,
                        template_coords=template_rect)  # Add more if you need to


def part_2a():
    num_particles = 3000  # Define the number of particles
    sigma_exp = 10.  # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 25.  # Define the value of sigma for the particles movement (dynamics)
    alpha = 0.92  # Set a value for alpha
#    template_rect = {'x': 540.8751, 'y': 390.1776, 'w': 60.5404, 'h': 90.0504}  # suggested template window (dict)
    # Define the template window values
    template_rect = {'x': 555.8751, 'y': 400.1776, 'w': 40.5404, 'h': 80.0504} 
    run_particle_filter(AppearanceModelPF,  # particle filter model class
                        os.path.join(input_dir, "pres_debate.mp4"),  # input video
                        template_rect,
                        {
                            'template': os.path.join(output_dir, 'ps7-2-a-1.png'),
                            22: os.path.join(output_dir, 'ps7-2-a-2.png'),
                            50: os.path.join(output_dir, 'ps7-2-a-3.png'),
                            160: os.path.join(output_dir, 'ps7-2-a-4.png')
                        },
                        num_particles=num_particles, sigma_exp=sigma_exp, sigma_dyn=sigma_dyn, alpha=alpha,
                        template_coords=template_rect)  # Add more if you need to


def part_2b():
    num_particles = 5000  # Define the number of particles
    sigma_exp = 5.  # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 40.  # Define the value of sigma for the particles movement (dynamics)
    alpha = 0.95  # Set a value for alpha
    template_rect = {'x': 555.8751, 'y': 400.1776, 'w': 40.5404, 'h': 80.0504}  # suggested template window (dict)

    run_particle_filter(AppearanceModelPF,  # particle filter model class
                        os.path.join(input_dir, "noisy_debate.mp4"),  # input video
                        template_rect,
                        {
                            'template': os.path.join(output_dir, 'ps7-2-b-1.png'),
                            22: os.path.join(output_dir, 'ps7-2-b-2.png'),
                            50: os.path.join(output_dir, 'ps7-2-b-3.png'),
                            160: os.path.join(output_dir, 'ps7-2-b-4.png')
                        },
                        num_particles=num_particles, sigma_exp=sigma_exp, sigma_dyn=sigma_dyn, alpha=alpha,
                        template_coords=template_rect)  # Add more if you need to


def part_3a():
    num_particles = 1000  # Define the number of particles
    sigma_chi = 10  # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 10  # Define the value of sigma for the particles movement (dynamics)
    hist_bins_num = 8
    template_rect = {'x': 320.8751, 'y': 175.1776, 'w': 103.5404, 'h': 129.0504}  #default

    run_particle_filter(MeanShiftLitePF,
                        os.path.join(input_dir, "pres_debate.mp4"),
                        template_rect,
                        {
                            'template': os.path.join(output_dir, 'ps7-3-a-1.png'),
                            28: os.path.join(output_dir, 'ps7-3-a-2.png'),
                            94: os.path.join(output_dir, 'ps7-3-a-3.png'),
                            171: os.path.join(output_dir, 'ps7-3-a-4.png')
                        },
                        num_particles=num_particles, sigma_exp=sigma_chi, sigma_dyn=sigma_dyn,
                        hist_bins_num=hist_bins_num,
                        template_coords=template_rect)  # Add more if you need to


def part_3b():
    num_particles = 5000  # Define the number of particles
    sigma_chi = 8. # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 25.  # Define the value of sigma for the particles movement (dynamics)
    hist_bins_num = 8  # Define the number of bins
    template_rect = {'x': 555.8751, 'y': 390.1776, 'w': 40.5404, 'h': 80.0504}  # Define the template window values

    run_particle_filter(MeanShiftLitePF,
                        os.path.join(input_dir, "pres_debate.mp4"),
                        template_rect,
                        {
                            'template': os.path.join(output_dir, 'ps7-3-b-1.png'),
                            22: os.path.join(output_dir, 'ps7-3-b-2.png'),
                            50: os.path.join(output_dir, 'ps7-3-b-3.png'),
                            160: os.path.join(output_dir, 'ps7-3-b-4.png')
                        },
                        num_particles=num_particles, sigma_exp=sigma_chi, sigma_dyn=sigma_dyn,
                        hist_bins_num=hist_bins_num,
                        template_coords=template_rect)  # Add more if you need to


if __name__ == '__main__':
    part_1a()
    part_1b()
    part_2a()
    part_2b()
    part_3a()
    part_3b()
