import {View, Text, Image, StyleSheet,TouchableOpacity} from 'react-native';
import React from 'react';
import {useNavigation} from '@react-navigation/native';
const Profile = () => {
  const navigation = useNavigation();
  return (
    <View style={styles.container}>
      <View style={{alignItems: 'center'}}>
        <Image
          source={require('../assets/house.jpg')}
          style={{width: 140, height: 140, borderRadius: 100, marginTop: 50}}
        />
      </View>
      <View style={styles.box}>
        <Image
          source={require('../assets/person.png')}
          style={styles.boxImage}
        />
        <Text style={styles.heading}>Arjun Babu</Text>
      </View>

      <View style={styles.box}>
        <Image
          source={require('../assets/email.png')}
          style={styles.boxImage}
        />
        <Text style={styles.heading}>npabhinand@gmail.com</Text>
      </View>

      <TouchableOpacity
        style={[styles.box,{backgroundColor:'#ff8717'}]}
        onPress={()=>navigation.navigate('Login')}
        >
        {/* <Image
          source={require('../assets/person.png')}
          style={styles.boxImage}
        /> */}
        <Text style={{color: '#FFFFFF',fontSize:20}}>Logout</Text>
      </TouchableOpacity>
    </View>
  );
};
const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  heading: {
    fontSize: 20,
    fontWeight: 'bold',
    marginLeft: 10,
  },
  box: {
    alignSelf: 'center',
    flexDirection: 'row',
    justifyContent: 'center',
    backgroundColor: '#fff',
    width: '90%',
    padding: 20,
    paddingBottom: 22,
    borderRadius: 10,
    shadowOpacity: 80,
    elevation: 15,
    marginTop: 40,
    // alignItems:'center'
  },
  boxImage: {
    width: 20,
    height: 20,
    marginTop: 5,
  },
});

export default Profile;
