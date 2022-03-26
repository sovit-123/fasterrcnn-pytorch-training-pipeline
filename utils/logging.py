import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def set_log(log_dir):
    logging.basicConfig(
        # level=logging.DEBUG,
        format='%(message)s',
        # datefmt='%a, %d %b %Y %H:%M:%S',
        filename=f"{log_dir}/train.log",
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # add the handler to the root logger
    logging.getLogger().addHandler(console)

def log(content, *args):
    for arg in args:
        content += str(arg)
    logger.info(content)

def coco_log(log_dir, stats):
    log_dict_keys = [
        'Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]',
        'Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ]',
        'Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ]',
        'Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]',
        'Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]',
        'Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]',
        'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ]',
        'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ]',
        'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]',
        'Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]',
        'Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]',
        'Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]',
    ]
    log_dict = {}
    # for i, key in enumerate(log_dict_keys):
    #     log_dict[key] = stats[i]

    with open(f"{log_dir}/train.log", 'a+') as f:
        f.writelines('\n')
        for i, key in enumerate(log_dict_keys):
            out_str = f"{key} = {stats[i]}"
            logger.debug(out_str) # DEBUG model so as not to print on console.
        logger.debug('\n'*2) # DEBUG model so as not to print on console.
    # f.close()
