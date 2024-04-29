import subprocess
import os
import queue
from threading import Thread


def worker(gpu_id, task_queue):
    """
    Continuously get a task from the queue, set the CUDA_VISIBLE_DEVICES environment variable, and execute the task.
    When the queue is empty, the worker will exit.
    """
    while True:
        try:
            # If the queue is empty, queue.Empty will be raised, and the worker will break the loop
            command = task_queue.get(timeout=3)  # You can adjust the timeout as needed
        except queue.Empty:
            return

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(7)

        # Execute the command
        process = subprocess.Popen(command, env=env, shell=True)
        process.wait()

        # Mark this task as done in the queue to allow another to be added if needed
        task_queue.task_done()


def execute_commands_on_gpus(commands, num_gpus=None):
    """
    Create a queue of commands, and have each GPU work through the queue.
    """
    # Query number of available GPUs
    if num_gpus is None:
        try:
            num_gpus = str(
                subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=gpu_name", "--format=csv,noheader"]
                ).decode("utf-8")
            ).count("\n")
            assert num_gpus > 0
        except Exception as e:
            print(
                f"An error occurred while querying the number of GPUs using nvidia-smi: {e}"
            )
            return

    # Create a queue for the commands
    command_queue = queue.Queue()
    for command in commands:
        command_queue.put(command)

    # Start a worker thread for each GPU
    threads = []
    for gpu_id in range(num_gpus):
        thread = Thread(target=worker, args=(gpu_id, command_queue))
        thread.start()
        threads.append(thread)

    # Wait for all tasks in the queue to be processed
    command_queue.join()

    # The commands are all done at this point, but the worker threads are likely idle and waiting for more tasks.
    # We'll end each thread by joining them.
    for thread in threads:
        thread.join()


def return_command(
    tag=None,
    prompt=None,
    called=0,
    use_freeu_until=0,
    lambda_corr=1000,
    use_corr_after=3000,
    use_corr_until=7000,
    azimuth_perturbation_start=10,
    azimuth_perturbation_end=30,
    edge_ksz=3,
    max_corr=60000,
    epipolar_thres=2,
    pretrained_path=None,
    visualize=0,
    visualize_name="placeholder",
):
    if pretrained_path is not None:
        command = f"""
            python3 launch.py \\
                --config configs/corr-mvdream-sd21-shading-v100-short.yaml \\
                --train \\
                --gpu 0 \\
                tag={tag} \\
                system.called={called} \\
                system.prompt_processor.prompt="{prompt}" \\
                system.guidance.use_freeu_until={use_freeu_until} \\
                system.guidance.called={called} \\
                system.loss.use_corr_loss=true \\
                system.loss.lambda_corr={lambda_corr} \\
                system.loss.use_corr_after={use_corr_after} \\
                system.loss.use_corr_until={use_corr_until} \\
                system.loss.azimuth_perturbation_start={azimuth_perturbation_start} \\
                system.loss.azimuth_perturbation_end={azimuth_perturbation_end} \\
                system.loss.edge_ksz={edge_ksz} \\
                system.loss.max_corr={max_corr} \\
                system.loss.epipolar_thres={epipolar_thres} \\
                resume="{pretrained_path}" \\
                system.loss.visualize={visualize} \\
                system.loss.visualize_name={visualize_name}
            """
    else:
        command = f"""
            python3 launch.py \\
                --config configs/corr-mvdream-sd21-shading-cfg-scheduling.yaml \\
                --train \\
                --gpu 0 \\
                tag={tag} \\
                system.called={called} \\
                system.prompt_processor.prompt="{prompt}" \\
                system.guidance.use_freeu_until={use_freeu_until} \\
                system.guidance.called={called} \\
                system.loss.use_corr_loss=true \\
                system.loss.lambda_corr={lambda_corr} \\
                system.loss.use_corr_after={use_corr_after} \\
                system.loss.use_corr_until={use_corr_until} \\
                system.loss.azimuth_perturbation_start={azimuth_perturbation_start} \\
                system.loss.azimuth_perturbation_end={azimuth_perturbation_end} \\
                system.loss.edge_ksz={edge_ksz} \\
                system.loss.max_corr={max_corr} \\
                system.loss.epipolar_thres={epipolar_thres} \\
                system.loss.visualize={visualize} \\
                system.loss.visualize_name={visualize_name}
            """

    return command


# Define the different commands you want to run (each as a string)

prompt_list = [
    # "a capybara wearing a top hat, low poly",
    # "a chimpanzee with a big grin",
    "a cute steampunk elephant",
    # "a DSLR photo of a covered wagon",
    # "a boy in mohawk hairstyle, head only, 4K, HD, raw",
    # "Wall-E, cute, render, super detailed, best quality, 4K, HD",
    # "a bichon frise wearing academic regalia",
    # "Samurai koala bear",
    # "Corgi riding a rocket",
    # "an orangutan holding a paint palette in one hand and a paintbrush in the other",
    # "a zoomed out DSLR photo of a pug made out of modeling clay",
    # "a zoomed out DSLR photo of a gummy bear driving a convertible",
    # "a yellow schoolbus",
    # "an astronaut riding a horse",
    # "a black cat wearing headphones listening to music, eyes closed",
    # "a knitted black rabbit wearing pink mufflers and green glasses",
]

commands_list = []
for prompt, name in zip(prompt_list, prompt_list):
    commands_list.append(
        return_command(
            tag="{}_{}_{}".format(name.replace(" ", "_"), 3000, 7000),
            prompt=prompt,
            called=0,
            lambda_corr=0,
            use_corr_after=3000,
            use_corr_until=7000,
        )
    )

# Execute the commands across the GPUs
execute_commands_on_gpus(commands_list)
