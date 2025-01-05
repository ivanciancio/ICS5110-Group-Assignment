import utilities as utils
import ui_engine as ui



def main():
    utils.init_session_variables()
    
    utils.load_and_cache_datasets()
    
    ui.build_ui()



main()